
# 动态并查集求连通分量
class mergeSet():
    def __init__(self, _initSize=100):
        self.initSize = _initSize
        # 初始化根节点
        self.father = []
        for i in range(self.initSize):
            self.father.append(i)
        # 初始化　“秩”
        self.rank = []
        for i in range(self.initSize):
            self.rank.append(1)

    # 路径压缩, 寻找根节点
    def find(self, u):
        if(u == self.father[u]):
            return u
        else:
            self.father[u] = self.find(self.father[u])
            return self.father[u]

    # 如果　i, j 之间有一条边, 把　i 和　j 加入同一连通分量
    def merge(self, i, j):
        x = self.find(i)
        y = self.find(j)
        if(x == y):
            return
        # "秩"　大的合并　"秩" 小的
        if(self.rank[x] > self.rank[y]):
            x, y = y, x
        self.father[x] = y
        self.rank[y] += self.rank[x]

    # 返回最后的聚类结果
    def cluster(self):
        clusters = {}
        for i in range(self.initSize):
            root = self.find(i)   # 一个连通分量内的所有点, 归一到同一个根节点
            self.father[i] = root
            if(root not in clusters.keys()):
                clusters[root] = [i]
            else:
                clusters[root].append(i)
        return clusters

    # 文件测试
    def loadFile(self, fileName):
        infile = open(fileName, 'r')
        assert(infile)
        oneLine = infile.readlines()
        for pair in oneLine:
            self.merge(int(pair[0]), int(pair[2]))
        infile.close()

    # 数据测试
    def loadData(self, pairs):
        for it in pairs:
            self.merge(int(pair[0]), int(pair[1]))


if __name__ == '__main__':
    one = mergeSet(10)
    one.loadFile('./cluster/merge.txt')
    clusters = one.cluster()
    print(clusters)
