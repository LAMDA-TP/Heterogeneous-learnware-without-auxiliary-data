import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


################
#  Color list  #
################

def get_color_list(type='default', size=None):
    """

    :param type: {'default','set1','set2','set3','Pastel1',colormap_name}
        colormap_name ref:
        https://www.matplotlib.org.cn/gallery/color/colormap_reference.html
        https://zhuanlan.zhihu.com/p/181615818
        Qualitative colormaps: https://www.cda.cn/discuss/post/details/5e721cdb69a1f41482604d96
    :return:
    """
    if type=='default':
        color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                      '#17becf']
    elif type=='set1':
        color_list=[]
        for i in range(9):
            color = plt.cm.Set1(i)
            color_standard = np.array(color).reshape(1, -1)
            color_list.append(color_standard)
    elif type=='set2':
        color_list=[]
        for i in range(8):
            color = plt.cm.Set2(i)
            color_standard = np.array(color).reshape(1, -1)
            color_list.append(color_standard)
    elif type=='set3':
        color_list=[]
        for i in range(12):
            color = plt.cm.Set3(i)
            color_standard = np.array(color).reshape(1, -1)
            color_list.append(color_standard)
    elif type=='Pastel1':
        color_list=[]
        for i in range(9):
            color = plt.cm.Pastel1(i)
            color_standard = np.array(color).reshape(1, -1)
            color_list.append(color_standard)
    elif type=='tab10':
        color_list=[]
        for i in range(10):
            color = plt.cm.tab10(i)
            color_standard = np.array(color).reshape(1, -1)
            color_list.append(color_standard)
    elif type=='tab20':
        color_list=[]
        for i in range(20):
            color = plt.cm.tab20(i)
            color_standard = np.array(color).reshape(1, -1)
            color_list.append(color_standard)
    else:
        if size is None:
            raise Exception('The size parameter is illegal!')
        cmap = mpl.cm.get_cmap(type, size)
        color_list = []
        for i in range(size):
            color = cmap.colors[i]
            color_standard = np.array(color).reshape(1, -1)
            color_list.append(color_standard)
    return color_list


class num2colorTransformer(object):

    def __init__(self, color_list):
        self.color_list=color_list

    def transform(self, numbers):
        if max(numbers)>len(self.color_list):
            assert Exception('Small color list!')
        colors = []
        for i in range(len(numbers)):
            colors.append(self.color_list[numbers[i]])
        return colors


def plot_color_list(color_list, color_list_name='', show_immediately=True):
    plt.figure()
    size=len(color_list)
    for i in range(size):
        plt.scatter(i,i,c=color_list[i],s=100)
    plt.xlim([-0.5,size])
    plt.ylim([-0.5, size])
    plt.title(color_list_name)
    if show_immediately:
        plt.show()

########################
#  RKME visualization  #
########################

class Contour_Grid():

    def __init__(self, bound_square):
        x_min, x_max, y_min, y_max = bound_square
        X, Y = np.mgrid[x_min:x_max:500j, y_min:y_max:500j]
        self.x = 500
        self.y = 500
        self.X = X
        self.Y = Y
        self.xy = np.c_[X.ravel(), Y.ravel()]

    def get_xy(self):
        return self.xy

    def reshape_to_grid(self, v):
        return np.reshape(v, (self.x, self.y))


def plot_density(rs, bound_square, cmap_str, levels=[0,0.6,0.7,0.8,0.9,0.95]):
    grid = Contour_Grid(bound_square)
    density = rs.eval(grid.xy)
    density = grid.reshape_to_grid(density)
    figure=plt.contour(grid.X, grid.Y, density, levels, cmap=cmap_str, alpha=1, zorder=1)
    plt.clabel(figure, inline=True, fontsize=8)
    return True


###########
#  t-SNE  #
###########


def my_tsne(X,y,title_str='',show_immediately=False):
    plt.figure()
    y = np.array(y)
    X = TSNE(n_components=2).fit_transform(X)
    legends=[]
    for y_temp in set(y):
        sub_index = np.argwhere(y == y_temp).squeeze()
        X_temp = X[sub_index, :]
        plt.scatter(X_temp[:, 0], X_temp[:, 1])
        legends.append(str(y_temp))
    plt.legend(legends)
    plt.title(title_str)
    if show_immediately:
        plt.show()


def scatter_with_legends(X,y):
    transformer=num2colorTransformer(get_color_list())
    X=np.array(X)
    y=np.array(y)
    fig=plt.figure(figsize=[4,4])
    if X.ndim!=2:
        ax = fig.add_subplot(1, 1, 1, projection='3d')
    legends = []
    for y_temp in set(y):
        sub_index = np.argwhere(y == y_temp).squeeze()
        X_temp = X[sub_index, :]
        if X.ndim==2:
            plt.scatter(X_temp[:, 0], X_temp[:, 1], c=transformer.transform([y_temp]*len(sub_index)))
        else:
            ax.scatter(X_temp[:, 0], X_temp[:, 1], X_temp[:, 2])
        legends.append(str(y_temp))
    plt.legend(legends)


if __name__=='__main__':
    type_list=['default','set1','set2','set3','Pastel1','tab10','tab20','viridis','plasma']
    for type in type_list:
        color_list=get_color_list(type,15)
        plot_color_list(color_list,type,show_immediately=False)
    plt.show()