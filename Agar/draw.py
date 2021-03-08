import matplotlib.pyplot as plt

def draw(x, y, x_label, y_label, title, save_path, label = None):

    plt.style.use('ggplot')
    if label is None:label = [None for i in range(len(y))]
    for i in range(len(y)):
        plt.plot(x, y[i], linewidth=1, label = label[i])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    #plt.title(title)
    plt.legend(loc='best', numpoints=1, fancybox=True)
    plt.savefig(save_path + '/' + title + '.png',dpi=120) # bbox_inches='tight')
    plt.close()
