import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pyecharts.charts import Scatter
from pyecharts import options as opts
from pyecharts.render import make_snapshot
from pyecharts.commons.utils import JsCode
from snapshot_selenium import snapshot
from sklearn.decomposition import PCA
# from model.gram import GRAM

def get_diagnosis_rep(params):
    """Get every the presentation for every diagnosis
    """    
    # read the type file
    types = pickle.load(open('../data/MIMIC_with_ancestors.types', 'rb'))
    rTypes = dict([(v,u) for u,v in types.items()])
    # count all disease number
    disease_num = []
    with open('../output/origin_gram/output_all_acc.txt', 'r') as f:
        lines = f.readlines()
        # data example
        # TypeNum:72, icd_code:D_428.0, Count:4581, avg_rank:5.822091, Accuracy:0.967693
        for line in lines:
            buf = line.strip().split(', ')
            icd_code = buf[1].split(':')[1]
            count = buf[2].split(':')[1]
            disease_num.append((types[icd_code], count))
    disease_num.reverse()

    # split the disease into 5 parts
    disease_divided = [[] for _ in range(10)]
    disease_divided_num = [[] for _ in range(10)]
    batch = len(disease_num) // 10
    for idx, item in enumerate(disease_num):
        for i in range(10):
            if i*batch <= idx < (i+1)*batch:
                disease_divided[i].append(item[0])
                disease_divided_num[i].append(float(item[1]))

    # get the middle num
    for idx, diseases in enumerate(disease_divided_num):
        print(f'Part {idx*10}_{(idx+1)*10}',diseases[batch//2])

    # get the representation for each disease
    preList = []
    for i in range(10):
        buf = params['W_emb'].cpu().numpy()
        # print(buf)
        # print(disease_divided)
        preList.append(buf[disease_divided[i]])

    pre = np.concatenate(preList)
    diseasesList = np.concatenate(disease_divided)
    diseasesList = np.reshape(diseasesList, (len(diseasesList), 1))
    print(pre.shape)
    print(diseasesList.shape)

    # PCA and random sample
    pca = PCA(n_components=2)
    pca_res = pca.fit_transform(pre)
    pca_res = np.concatenate((pca_res, diseasesList), axis=1)
    print(pca_res.shape)

    # Use echarts to visualize
    np.random.seed(4396)
    scatters = []
    tool_js = """function (param) {return param.seriesName + ' — ' +param.data[3]+'<br/>' 
                +'x轴坐标： '+param.data[0]+' <br/>'
                +'y轴坐标： '+param.data[1];}"""
    
    for i in range(10):
        part_pre = pca_res[i*batch:(i+1)*batch]
        np.random.shuffle(part_pre)
        data = part_pre.tolist()[:20]
        scatter = (Scatter()
                   .add_xaxis([buf[0] for buf in data])
                   .add_yaxis(str(i*10) + '_' + str((i+1)*10), [[buf[1],buf[0],rTypes[buf[2]]] for buf in data],
                              label_opts=opts.LabelOpts(is_show=False))
                   .set_global_opts(yaxis_opts=opts.AxisOpts(name='y轴', type_="value"), 
                                    xaxis_opts=opts.AxisOpts(name='x轴', type_="value"),
                                    tooltip_opts = opts.TooltipOpts(formatter=JsCode(tool_js)))
                   )
        scatters.append(scatter)
        # plt.scatter(part_pre[:, 0], part_pre[:, 1], label=str(i*10) + '_' + str((i+1)*10))
    # plt.legend(ncol=4)
    # plt.savefig(model_path + '_random_patient_pre.jpg')
    # plt.show()
    scatter1 = scatters[0]
    for i in range(1,10):
        scatter1.overlap(scatters[i])
    # scatter1.set_global_opts(title_opts=opts.TitleOpts(title="疾病分布图"))
    scatter1.render()



def read_params(model_path):
    # get the best model through log
    with open(model_path+'.log') as f:
        line = f.readlines()[-2]
        best_epoch = line.split(',')[0].split(':')[1]
    
    params = torch.load(model_path + '.' + best_epoch)
    return params



def params_change(params, target_diseases):
    types = pickle.load(open('/home/wurui/GRAM_pytorch/data/MIMIC_with_ancestors.types', 'rb'))
    emb1 = params['W_emb'][types[target_diseases[0]]]
    emb2 = params['W_emb'][types[target_diseases[1]]]
    params['W_emb'][types[target_diseases[0]]] = emb1 + 0.5
    return params
        
    


if __name__ == "__main__":
    parameters = read_params('../output/origin_gram/output')
    parameters = params_change(parameters, ['D_707.19', 'D_996.82'])
    # torch.save(parameters, '../output/origin_gram/output_changed')
    get_diagnosis_rep(parameters)
