import re

end_layer_map = []
end_layer_list = []
early_exit_cumm = 0
early_exit_cumm_list = []
early_exit_cumm_list_large= []
early_exit_cumm_map = []
n_sample = 0
f = open('batch_30_2.log', 'r')

for line in f.readlines():
    '''if n_sample == 30:
        n_sample = 0
        end_layer_map.append(end_layer_list)
        early_exit_cumm_map.append(early_exit_cumm_list)
        end_layer_list = []
        early_exit_cumm_list = []'''

    if 'server idle!' in line:
        end_layer_list.append(-1)
        early_exit_cumm_list.append(early_exit_cumm)
        early_exit_cumm_list_large.append(early_exit_cumm)
    elif 'end idx:' in line:
        end_layer_list.append(int(re.search(r'\d+', line).group()))

    if 'is early:  True' in line:
        early_exit_cumm = early_exit_cumm + 1
        early_exit_cumm_list.append(early_exit_cumm)
        early_exit_cumm_list_large.append(early_exit_cumm)
    if 'is early:  False' in line:
        early_exit_cumm_list.append(early_exit_cumm)
        early_exit_cumm_list_large.append(early_exit_cumm)
    if 'early oom!' in line:
        early_exit_cumm_list.append(early_exit_cumm)
        early_exit_cumm_list_large.append(early_exit_cumm)

    if 'tatol time:' in line:
        n_sample = n_sample + 1
        end_layer_map.append(end_layer_list)
        early_exit_cumm_map.append(early_exit_cumm_list)
        end_layer_list = []
        early_exit_cumm_list = []



f.close
print('map: ', end_layer_map)
print('early: ', early_exit_cumm_list_large)
#print('early cumm: ', early_exit_cumm_list)
print('size: ', len(early_exit_cumm_list_large))

for e_list in early_exit_cumm_map:
    print(e_list)
    print(len(e_list))


'''
f = open('test_resnet34_cifar10_runs20.log', 'r')

for line in f.readlines():
    if 'block 1:' in line:
        print(line)
f.close
'''