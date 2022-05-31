import copy

import numpy as np

branch_index_list = ['branch0', 'branch1', 'branch2', 'branch3', 'branch4', 'branch5', 'branch6', 'branch8',
                     'branch9', 'branch10', 'branch11', 'branch12', 'branch13', 'branch14', 'branch15', 'branch16',
                     'branch17', 'branch18', 'branch19', 'branch20', 'branch21', 'branch22', 'branch23', 'branch24',
                     'branch25', 'branch26', 'branch27', 'branch28', 'branch29', 'branch30', 'branch32', 'branch33',
                     'branch34', 'branch36', 'branch37', 'branch38', 'branch39', 'branch40', 'branch41', 'branch42',
                     'branch43', 'branch44', 'branch45', 'branch46', 'branch47', 'branch48', 'branch49', 'branch51',
                     'branch52', 'branch53', 'branch54', 'branch55', 'branch56', 'branch57', 'branch58', 'branch59',
                     'branch60', 'branch61', 'branch62', 'branch63', 'branch64', 'branch65', 'branch66', 'branch67',
                     'branch68', 'branch69', 'branch70', 'branch71', 'branch72', 'branch73', 'branch74', 'branch75',
                     'branch76', 'branch77', 'branch78', 'branch79', 'branch80', 'branch81', 'branch82', 'branch83',
                     'branch84', 'branch85', 'branch86', 'branch87', 'branch88', 'branch89', 'branch90', 'branch91',
                     'branch93', 'branch95', 'branch96', 'branch97', 'branch98', 'branch99', 'branch100', 'branch102',
                     'branch103', 'branch104', 'branch105', 'branch107', 'branch108', 'branch109', 'branch110',
                     'branch111',
                     'branch112', 'branch113', 'branch114', 'branch115', 'branch116', 'branch117', 'branch118',
                     'branch119',
                     'branch120', 'branch121', 'branch122', 'branch123', 'branch124', 'branch125', 'branch127',
                     'branch128',
                     'branch129', 'branch130', 'branch131', 'branch132', 'branch133', 'branch134', 'branch135',
                     'branch136',
                     'branch137', 'branch138', 'branch139', 'branch140', 'branch141', 'branch142', 'branch143',
                     'branch144',
                     'branch145', 'branch146', 'branch147', 'branch148', 'branch149', 'branch150', 'branch151',
                     'branch152',
                     'branch153', 'branch154', 'branch155', 'branch156', 'branch157', 'branch158', 'branch159',
                     'branch160',
                     'branch161', 'branch162', 'branch163', 'branch164', 'branch165', 'branch166', 'branch167',
                     'branch168',
                     'branch169', 'branch170', 'branch171', 'branch172', 'branch173', 'branch174', 'branch175',
                     'branch176',
                     'branch177', 'branch178', 'branch179', 'branch180', 'branch181', 'branch182', 'branch183',
                     'branch184',
                     'branch185', 'branch186', 'branch187', 'branch188', 'branch189', 'branch190', 'branch191',
                     'branch192',
                     'branch193']

def dispatching_necessity_evaluation(obs,p,adj_p,fiR,fiL,ndis,omigaP=1.5):
    necessities=[]
    for i in range(len(p)):
        necessity=get_necessity(i,obs,p,adj_p,fiR,fiL,omigaP)
        necessities.append(necessity)

    # print(necessities)
    max_indexs=[]
    necessities_cal= copy.deepcopy(necessities)
    for i in range(ndis):
        # max_ne = max(necessities_cal)
        max_index = necessities_cal.index(max(necessities_cal))
        necessities_cal[max_index]=-1000
        max_indexs.append(max_index)

    finall_adj_p=copy.deepcopy(adj_p)
    for i in range(len(p)):
        if i not in max_indexs:
            finall_adj_p[i]=0
    return finall_adj_p



def get_necessity(i,obs,next_p,adj_p,fiR,fiL,omigaP):
    adj_p_abs=abs(np.array(adj_p))
    rho = line_over_flow_reward_sep(obs,i)
    
    max_p = obs.action_space['adjust_gen_p'].high[i]+obs.gen_p[i]
    if max_p !=0:
	    a1 = omigaP * (adj_p_abs[i]-min(adj_p_abs))/(max(adj_p_abs)-min(adj_p_abs))
	    a2 = (fiR - next_p[i] / max_p) * adj_p[i]
	    a3 = (fiL - max(rho)) * adj_p[i]
#	    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
#	    print(i)
#	    print(obs.action_space['adjust_gen_p'].high[i])
#	    print(a1,a2,a3)
	    a = a1 + a2 + a3
	    return a
    else:
    	    return -100
	  

def line_over_flow_reward_sep(obs,gen_i):
    reorder=sorted(obs.unnameindex.items(),key=lambda d:d[1])#"gen_10021":0,1###!!!!!!
    reorder_unnameindex=dict(reorder)
#    print(obs.bus_gen)
#    print(reorder_unnameindex)
#    print([key for key in reorder_unnameindex])
    gen_names= [key for key in reorder_unnameindex]

    #for key in reorder_unnameindex:
    branch_list = obs.bus_branch[get_bus(obs.bus_gen,gen_names[gen_i])]
    num_branch=len(branch_list)
    branch_list_name = []
    for ele in range(0, num_branch):
        if "or" in branch_list[ele]:
            str = branch_list[ele].rstrip("_or")
            branch_list_name.append(str)
        if "ex" in branch_list[ele]:
            str = branch_list[ele].rstrip("_ex")
            branch_list_name.append(str)
    branch_list_index = []
    if branch_list_name != []:
        # continue
    # else:
        for ele in range(0, num_branch):
            # if branch_list_name[ele]=="branch31" or branch_list_name[ele]=="branch101" or branch_list_name[ele]=="branch7" or branch_list_name[ele]=="branch94" or branch_list_name[ele]=="branch106":
            #     continue
            # else:
            #     branch_list_index.append(branch_index_list.index(branch_list_name[ele]))
            if branch_list_name[ele] in branch_index_list:
                #branch_list_index.append(ele)
                branch_list_index.append(branch_index_list.index(branch_list_name[ele]))
    rho_list=[]
    if branch_list_name==[]:
        rho_list.append(0)
    else:
        for ele in range(0,len(branch_list_index)):
            rho_list.append(obs.rho[branch_list_index[ele]])
    return rho_list


def get_bus(dic, value):
    list1 = []
    list1.append(value)
    key = list(dic.keys())[list(dic.values()).index(list1)]
    return key

