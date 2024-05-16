import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


label2color_dict = {
    0: [255,   0,   0],
    1: [  0,   0, 255],
    2: [  0, 100, 100],
    3: [150,  40,   0],
    4: [170,  20, 100],
    5: [255, 248, 220],  # cornsilk
    6: [100, 149, 237],  # cornflowerblue
    7: [102, 205, 170],  # mediumAquamarine
    8: [205, 133,  63],  # peru
    9: [160,  32, 240],  # purple
    10:[255,  64,  64],  # brown1
}

isprs_cls_label = {
    1: 'Building',
    0: 'Background', 
}


def tSNE(feat, label, mask, n_classes, len_s, title):
    def plot_embedding(feat, label, mask, title):
        x_min, x_max = np.min(feat, 0), np.max(feat, 0)
        feat = (feat - x_min) / (x_max - x_min)
        
        plt.rcParams['savefig.dpi'] = 600
        plt.rcParams['figure.dpi'] = 600 
        fig = plt.figure()

        for i in range(feat.shape[0]):
            # label limitation
            label_i = int(label[i])
            mask_i = mask[i]

            # feat_s
            if i < len_s:
                size = 2
                if mask_i == 0:
                    size *= 5
                color = label2color_dict[label_i]
                color =  [j/255.0 for j in color] 
                marker = '.'
            # feat_t
            else:
                size = 2
                if mask_i == 0:
                    size *= 5
                color = label2color_dict[label_i]
                color =  [j/255.0 for j in color] 
                marker = 'x'
           
            # plot 
            plt.scatter(feat[i, 0], feat[i, 1], s=size, color=(color[0], color[1], color[2]), marker=marker, linewidths=0.3)

        for i in range(0, n_classes):
            color = label2color_dict[i]
            color =  [j/255.0 for j in color] 
            label = isprs_cls_label[i]
            plt.plot([], [], color=color, linewidth=5, linestyle="-", label=label)
        plt.legend(loc='upper right', fontsize=6)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(title)

    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(feat)
    plot_embedding(result, label, mask, title)
    # plt.show(fig)