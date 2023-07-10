import torch
import util.utils as utils
import temporalGraph
import numpy as np
from definitions import FRAMES_DIR
import os

def test(model, model_used, loss, test_loader, reference_frame, obj_predicted, viz, buffer_size, DEVICE, EXIT_TOKEN, N, SIMILARITY_THRESHOLD, T, LOOK_FORWARD, OBJECTS_ALLOWED, STRIDE):    
    print("Testing")
    temporal_graph = temporalGraph.TemporalGraph(DEVICE, buffer_size, OBJECTS_ALLOWED, N, STRIDE)
    data_loader_test = iter(test_loader)
    print("O dataset de teste tem: " + str(len(data_loader_test)) + " amostras")
    with torch.no_grad():
        model.eval()

        loss_mean = 0
        for i in range(len(data_loader_test)):
            print(i)
            input = next(data_loader_test)
            folder_index = input[1][0]
            sample_index = input[2][0]
            input = input[0]       # [1,16,240,320,3]   tensor.shape

            if model_used == 'i3d':
                # I3D recieves: (1, 3, 16, 224, 224), 16 is the segment lenght. So we have to adjust
                shape_ = input.shape
                input_frames = input.view(shape_[0], shape_[4], shape_[1], shape_[2], shape_[3]).type(torch.FloatTensor).to(DEVICE)

                # In the pretext task, we have to predict a future frame, so we need here just the first T frames. After, we will predicrt the LOOK_FORWARDº frame
                input_frames = input_frames[:, :, 0:T, :, :]
  

            input = np.squeeze(input)

            if model_used == 'i3d':
                T = LOOK_FORWARD

            cache_folder = "cache_pt_task/i3d/test/T="+str(T)+"-N="+str(N)+"/"
            data_path = os.path.join(FRAMES_DIR, cache_folder, str(folder_index.cpu().numpy()), str(sample_index.cpu().numpy())+"_data.npy")
            has_cache = False
            if os.path.exists(data_path):
                has_cache = True
                test_loader.has_cache = True
            else:
                has_cache = False
                
            if not has_cache:
                adj_mat, bbox_fea_list, box_list, score_list = temporal_graph.frames2temporalGraph(input, folder_index, sample_index)

                #SIMILARITY_THRESHOLD = 0.65#0.73

                graph = utils.calculeTargetAll(adj_mat, bbox_fea_list, box_list, score_list, reference_frame, SIMILARITY_THRESHOLD, T, N)


                # If in the first frame there is no object detected, so we have nothing to do here
                # The number of detected objects may be less than N. In this case we have nothing to do here
                #if len(bbox_fea_list[reference_frame][obj_predicted]) < N:
                #    print("continuando")
                #    continue       # Continue

                data, object_path = utils.calculeTarget(graph, score_list, bbox_fea_list, box_list, reference_frame, obj_predicted, DEVICE, EXIT_TOKEN, SIMILARITY_THRESHOLD, T, N)

                path = os.path.join(FRAMES_DIR, cache_folder, str(folder_index.cpu().numpy()))
                os.makedirs(path, exist_ok=True)

                if data != -1:
                    data = [data[0].cpu().numpy(), data[1].cpu().numpy()]
                
                np.save(data_path, data)
            else:                
                print("Ok, temos cache, vamos carregar")
                data = np.load(data_path, allow_pickle=True)

            if data == -1:
                print("Continuing because there aren't a object in the first frame ")
                continue

            data = [torch.from_numpy(data[0]).to(DEVICE), torch.from_numpy(data[1]).to(DEVICE)]

            # i3d needs frames in input. Otherwise, we need put the bondingbox and bbox features in input. 'input' var initially is the frames.
            
            input, target = data       
            if model_used == 'i3d':
                print("POOOOOOOOORRA")
                print(input_frames.shape)
                input_frames = {'frames': input_frames}
                output = model(input_frames)
            else:
                output = model(input)

            loss_ = loss(output, target)

            loss_mean += loss_.item()

        loss_mean = loss_mean / len(data_loader_test)
        print("Mean loss: ", str(loss_mean))
        try:
            viz.plot_lines('test_loss', loss_mean)
        except:
            print("Visdom not on")
        return loss_mean
