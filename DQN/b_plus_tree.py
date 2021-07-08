import numpy as np

class IndexNode:
    def __init__(self, num, index, child, parent):
        self.num = num
        self.index = index
        self.child = child
        self.parent = parent
        self.isLeaf = False
class LeafNode:
    def __init__(self, num, index, data, parent, nextLeaf):
        self.num = num
        self.index = index
        self.data = data
        self.parent = parent
        self.nextLeaf = nextLeaf
        self.isLeaf = True

class BPlusTree(object):
    node_pointer = 0
    def __init__(self, m):
        self.m = m
        self.maxkey = m - 1
        self.minkey = np.ceil(m / 2) - 1
        self.root = LeafNode(0, np.zeros(m), [None for i in range(m)], None, None)

    def find(self, p, index, num):
        locate = num
        for i in range(num):
            if p < index[i]:
                locate = i
                break
        return locate

    def insertNode(self, root, p, data):
        if root.isLeaf:
            locate = self.find(p, root.index, root.num)
            print('root.num, locate', root.num, locate)
            for i in range(root.num - 1, locate - 1, -1):
                root.index[i + 1] = root.index[i]
                root.data[i + 1] = root.data[i]
            root.index[locate] = p
            root.data[locate] = data
            root.num += 1
            print('root.index', root.index)
            return root
        else:
            return self.insertNode(root.child[self.find(p, root.index, root.num)], p, data)

    def add(self, p, data):
        addNode = self.insertNode(self.root, p, data)
        if addNode.num > self.maxkey:
            self.splitLeafNode(addNode)

    def splitLeafNode(self, lNode):
        splitLocate = (int)(lNode.num / 2)
        print('splitLocate', splitLocate)
        valueIndex = lNode.index[splitLocate]
        parent = lNode.parent
        if parent == None:
            print('None')
            parent = IndexNode(0, np.zeros(self.m + 1), [None for i in range(m + 1)], None)
            self.root = parent
        newlNode = LeafNode(0, np.zeros(self.m), [None for i in range(m)], None, None)
        insert = 0
        for i in range(splitLocate, lNode.num):
            newlNode.index[insert] = lNode.index[i]
            lNode.index[i] = 0
            newlNode.data[insert] = lNode.data[i]
            lNode.data[i] = None
            insert += 1
        lNode.num = splitLocate
        newlNode.num = insert
        lNode.parent = parent
        newlNode.parent = parent

        newlNode.nextLeaf = lNode.nextLeaf
        lNode.nextLeaf = newlNode

        parlocate = self.find(valueIndex, parent.index, parent.num)

        for i in range(parent.num - 1, parlocate - 1, -1):
            parent.index[i + 1] = parent.index[i]
            parent.child[i + 2] = parent.child[i + 1]

        parent.index[parlocate] = valueIndex
        parent.num += 1
        parent.child[parlocate] = lNode
        parent.child[parlocate + 1] = newlNode

        if parent.num >self.maxkey:
            self.splitIndexNode(parent)

    def splitIndexNode(self, iNode):
        splitLocate = (int)(iNode.num / 2)
        valueIndex = iNode.index[splitLocate]
        parent = iNode.parent
        if parent == None:
            parent = IndexNode(0, np.zeros(self.m + 1), [None for i in range(m + 1)], None)
            self.root = parent
        newInNode = IndexNode(0, np.zeros(self.m + 1), [None for i in range(m + 1)], None)
        insert = 0
        for i in range(splitLocate + 1, iNode.num, 1):
            newInNode.index[insert] = iNode.index[i]
            iNode.index[i] = 0
            newInNode.child[insert] = iNode.child[i]
            iNode.child[i] = None
            midNode = newInNode.child[insert]
            midNode.parent = newInNode
            insert += 1
        newInNode.child[insert] = iNode.child[iNode.num]
        # if newInNode.child[insert].isLeaf:
        midNode = newInNode.child[insert]
        midNode.parent = newInNode
        iNode.child[iNode.num] = None

        iNode.index[splitLocate] = 0
        newInNode.num = insert
        iNode.num = splitLocate
        newInNode.parent = parent
        iNode.parent = parent
        parLocate = self.find(valueIndex, parent.index, parent.num)

        for i in range(parent.num - 1, parLocate - 1, -1):
            parent.index[i + 1] = parent.index[i]
            parent.child[i + 2] = parent.child[i + 1]

        parent.index[parLocate] = valueIndex
        parent.num += 1
        parent.child[parLocate] = iNode
        parent.child[parLocate + 1] = newInNode

        if parent.num >self.maxkey:
            self.splitIndexNode(parent)

    def query(self, p, root):
        loc = self.find(p, root.index, root.num)
        if root.isLeaf:
            if root.index[loc - 1] == p:
                # succeeded
                print('Query Succeeded!', root.index[loc - 1])
                return root.data[loc - 1]
            else:
                # query faild, find the nearest one, which having three situations: head, body, tail
                # the first
                nstloc = loc - 1


                return root.data[nstloc]
        else:
            return self.query(p, root.child[loc])

def dfs(root):
    for i in range(root.num):
        print('index', root.index[i])
    if root.isLeaf:
        for i in range(root.num):
            print('data', root.data[i])
        return
    for i in range(root.num + 1):
        dfs(root.child[i])

def printLeafNode(root):
    while(root.isLeaf == False):
        root = root.child[0]
    while(root != None):
        for i in range(root.num):
            print('index and data', root.index[i], root.data[i])
        root = root.nextLeaf

if __name__ == "__main__":

    a = [3, 89, 45, 62, 15, 7, 26, 59, 48, 65, 32, 16, 99, 41]
    n = len(a)
    m = 4
    bPlusTree = BPlusTree(m)
    for i, val in enumerate(a):
        bPlusTree.add(val, i)
    root = bPlusTree.root
    dfs(root)
    # printLeafNode(root)
    print(bPlusTree.query(a[12], root))
