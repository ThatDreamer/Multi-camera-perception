import numpy as np
from queue import Queue
class IndexNode:
    """
    the Index Node Class
    Attributes:
        num: the current number of keywords
        index: the index array
        child: the child node array
        parent: the parent node of node
        isLeaf: is leaf node or not
    """
    def __init__(self, num, index, child, parent):
        self.num = num
        self.index = index
        self.child = child
        self.parent = parent
        self.isLeaf = False
class LeafNode:
    """
    Leaf Node Class
    Attributes:
        num: the current number of keywords
        index: the index array
        data: the data array
        parent: the parent node of node
        nextLeaf: the next leaf node
        isLeaf: is leaf node or not
    """
    def __init__(self, num, index, data, parent, nextLeaf):
        self.num = num
        self.index = index
        self.data = data
        self.parent = parent
        self.nextLeaf = nextLeaf
        self.isLeaf = True

class BPlusTree(object):
    """
    the B+Tree Class
    Attributes:
        m: the order of tree
        maxkey: the maximum number of keywords
        minkey: the minimum number of keywords
        root: the initial root node
    """
    def __init__(self, m):
        self.m = m
        self.maxkey = m - 1
        self.minkey = np.ceil(m / 2) - 1
        self.root = LeafNode(0, np.zeros(m), [None for i in range(m)], None, None)

    def find(self, p, index, num):
        """
        find the location in the index array
        :param p: the index to find
        :param index: the index array
        :param num: the number of keywords in the index array
        :return: the location of p
        """
        locate = num
        for i in range(num):
            if p < index[i]:
                locate = i
                break
        return locate

    def insertToNode(self, root, p, data):
        """
        insert the index p and data into leaf node
        :param root: the node currently traversed
        :param p: the index to insert
        :param data: the data to insert
        :return: the inserted leaf node
        """
        if root.isLeaf:
            locate = self.find(p, root.index, root.num)
            # print('root.num, locate', root.num, locate)
            # make a space for insertion
            for i in range(root.num - 1, locate - 1, -1):
                root.index[i + 1] = root.index[i]
                root.data[i + 1] = root.data[i]
            # insert
            root.index[locate] = p
            root.data[locate] = data
            root.num += 1
            # print('root.index', root.index)
            return root
        else:
            return self.insertToNode(root.child[self.find(p, root.index, root.num)], p, data)

    def add(self, p, data):
        """
        add the index p and data
        :param p: the index p to add
        :param data: the data to add
        :return: null
        """
        addNode = self.insertToNode(self.root, p, data)
        # determine whether to split
        if addNode.num > self.maxkey:
            self.splitLeafNode(addNode)

    def splitLeafNode(self, lNode):
        """
        split the leaf node
        :param lNode: the leaf node to be split
        :return: null
        """
        # calculate the split location
        splitLocate = (int)(lNode.num / 2)
        # print('splitLocate', splitLocate)
        # the new index
        valueIndex = lNode.index[splitLocate]
        parent = lNode.parent
        if parent == None:
            # print('None')
            parent = IndexNode(0, np.zeros(self.m + 1), [None for i in range(m + 1)], None)
            self.root = parent
        newlNode = LeafNode(0, np.zeros(self.m), [None for i in range(m)], None, None)

        # move the index and data to new leaf node
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
        newlNode.parent = parent  # pay attention to connecting the new node and the parent node

        # connect the new and old node
        newlNode.nextLeaf = lNode.nextLeaf
        lNode.nextLeaf = newlNode

        # insert the new index to parent index node
        parlocate = self.find(valueIndex, parent.index, parent.num)
        for i in range(parent.num - 1, parlocate - 1, -1):
            parent.index[i + 1] = parent.index[i]
            parent.child[i + 2] = parent.child[i + 1]
        parent.index[parlocate] = valueIndex
        parent.num += 1
        parent.child[parlocate] = lNode
        parent.child[parlocate + 1] = newlNode  # pay attention to connecting the new node and the parent node

        # determine whether to split
        if parent.num >self.maxkey:
            self.splitIndexNode(parent)

    def splitIndexNode(self, iNode):
        """
        split the index node
        :param iNode: the index node to be split
        :return: null
        """
        # calculate the split location
        splitLocate = (int)(iNode.num / 2)
        # the new index to move to the parent index node
        valueIndex = iNode.index[splitLocate]
        parent = iNode.parent
        if parent == None:
            parent = IndexNode(0, np.zeros(self.m + 1), [None for i in range(m + 1)], None)
            self.root = parent
        newInNode = IndexNode(0, np.zeros(self.m + 1), [None for i in range(m + 1)], None)

        # move the index and child to new index node which is different from the last situation
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
        midNode = newInNode.child[insert]
        midNode.parent = newInNode
        iNode.child[iNode.num] = None
        iNode.index[splitLocate] = 0
        newInNode.num = insert
        iNode.num = splitLocate
        newInNode.parent = parent  # pay attention to connecting the new node and the parent node
        iNode.parent = parent

        # insert the new index to parent index node
        parLocate = self.find(valueIndex, parent.index, parent.num)
        for i in range(parent.num - 1, parLocate - 1, -1):
            parent.index[i + 1] = parent.index[i]
            parent.child[i + 2] = parent.child[i + 1]
        parent.index[parLocate] = valueIndex
        parent.num += 1
        parent.child[parLocate] = iNode
        parent.child[parLocate + 1] = newInNode  # pay attention to connecting the new node and the parent node

        # determine whether to split
        if parent.num >self.maxkey:
            self.splitIndexNode(parent)

    def query(self, p, root):
        """
        the query function
        :param p: the index p to query
        :param root: the current node
        :return: the data of index p
        """
        loc = self.find(p, root.index, root.num)
        if root.isLeaf:
            if root.index[loc - 1] == p:
                # succeeded
                print('Query Succeeded!', root.index[loc - 1])
                return root.data[loc - 1]
            else:
                # query faild, return the nearest one
                nstloc = loc - 1
                return root.data[nstloc]
        else:
            return self.query(p, root.child[loc])

    def mergeLeafNode(self, lNode1, lNode2, faLoc):
        """
        merge the leaf node1 and node2
        :param lNode1: the leaf node1
        :param lNode2: the leaf node2
        :param faLoc: the location in parent node
        :return: null
        """
        # connect the leaf node1 and the the next leaf node of the leaf node2
        lNode1.nextLeaf = lNode2.nextLeaf
        # move the index and data of leaf node2 to the leaf node1
        for i in range(lNode1.num, lNode1.num + lNode2.num):
            lNode1.index[i] = lNode2.index[i - lNode1.num]
            lNode1.data[i] = lNode2.data[i - lNode1.num]
        lNode1.num += lNode2.num
        lNode2.num = 0
        lNode2.parent = None

        # delete the index and child from parent index node
        parent = lNode1.parent
        for i in range(faLoc - 1, parent.num - 1):
            parent.index[i] = parent.index[i + 1]
            parent.child[i + 1] = parent.child[i + 2]
        parent.index[parent.num - 1] = 0
        parent.child[parent.num] = None
        parent.num -= 1

        # determine whether it is the root node
        if parent.num < self.minkey and parent.parent == None:
            # print('lNode1', lNode1.index)
            if parent.num == 0:
                self.root = lNode1
                self.root.parent = None
            # print(self.root.index)
        # determine whether to merge
        elif parent.num < self.minkey:
            # print('parent.index[0]', parent.index[0])
            # find the location of the index to delete in the parent node of the node
            faLoc = self.find(parent.index[0], parent.parent.index, parent.parent.num)
            # maintain the tree
            self.maintainTree(parent, faLoc)

    def mergeIndexNode(self, iNode1, iNode2, faLoc):
        """
        merge the index node1 and node2
        :param iNode1: the index node1
        :param iNode2: the index node2
        :param faLoc: the location in parent node
        :return: null
        """
        # move the index of parent index node to the index node1
        tempIndex = iNode1.parent.index[faLoc - 1]
        iNode1.index[iNode1.num] = tempIndex
        # move the index and child of the index node2 to the index node1
        for i in range(iNode1.num + 1, iNode1.num + iNode2.num + 1):
            iNode1.index[i] = iNode2.index[i - iNode1.num - 1]
            iNode1.child[i] = iNode2.child[i - iNode1.num - 1]
            midNode = iNode2.child[i - iNode1.num - 1]  # pay attention to connecting the child node with the index node1
            midNode.parent = iNode1
        iNode1.num += (iNode2.num + 1)  # the 1 is the index from parent index node
        iNode1.child[iNode1.num] = iNode2.child[iNode2.num]
        midNode = iNode2.child[iNode2.num]
        midNode.parent = iNode1
        iNode2.num = 0
        iNode2.parent = None

        # delete the index and child from parent index node
        parent = iNode1.parent
        for i in range(faLoc - 1, parent.num - 1):
            parent.index[i] = parent.index[i + 1]
            parent.child[i + 1] = parent.child[i + 2]
        parent.index[parent.num - 1] = 0
        parent.child[parent.num] = None
        parent.num -= 1

        # determine whether it is the root node
        if parent.num < self.minkey and parent.parent == None:
            if parent.num == 0:
                self.root = iNode1
                self.root.parent = None
        # determine whether to merge
        elif parent.num < self.minkey:
            # find the location of the index to delete in the parent node of the node
            faLoc = self.find(parent.index[0], parent.parent.index, parent.parent.num)
            # maintain the tree
            self.maintainTree(parent, faLoc)
    def eraseFromNode(self, root, p, faLoc):
        """
        erase the index p and data from leaf node
        :param root: the node currently traversed
        :param p: the index p to erase
        :param faLoc: the location in parent node
        :return: the erased leaf node and the location in parent node
        """
        loc = self.find(p, root.index, root.num)
        if root.isLeaf:
            if root.index[loc - 1] == p:
                # erase the index p and data
                for i in range(loc - 1, root.num - 1):
                    root.index[i] = root.index[i + 1]
                    root.data[i] = root.data[i + 1]
                root.index[root.num - 1] = 0
                root.data[root.num - 1] = None
                root.num -= 1
                return root, faLoc
            else:
                print("Query Faild!")
        else:
            return self.eraseFromNode(root.child[loc], p, loc)

    def borrowFromLeftLeafNode(self, leftNode, root, faLoc):
        """
        borrow from the left leaf node
        :param leftNode: the left leaf node
        :param root: the node currently traversed
        :param faLoc: the location in parent node
        :return: null
        """
        tempIndex = leftNode.index[leftNode.num - 1]
        tempData = leftNode.data[leftNode.num - 1]
        root.parent.index[faLoc - 1] = tempIndex
        _, _ = self.eraseFromNode(leftNode, tempIndex, -1)
        _ = self.insertToNode(root, tempIndex, tempData)

    def borrowFromLeftIndexNode(self, leftNode, root, faLoc):
        """
        borrow from the left index node
        :param leftNode: the left index node
        :param root: the node currently traversed
        :param faLoc: the location in parent node
        :return: null
        """
        tempIndex = leftNode.index[leftNode.num - 1]
        tempChild = leftNode.child[leftNode.num]
        root.parent.index[faLoc - 1] = tempIndex
        leftNode.index[leftNode.num - 1] = 0
        leftNode.child[leftNode.num] = None
        leftNode.num -= 1

        # need to move the child node
        parlocate = 0
        for i in range(root.num - 1, parlocate - 1, -1):
            root.index[i + 1] = root.index[i]
            root.child[i + 2] = root.child[i + 1]

        root.index[parlocate] = tempIndex
        root.num += 1
        root.child[parlocate + 1] = root.child[parlocate]
        root.child[parlocate] = tempChild
        tempChild.parent = root



    def borrowFromRightLeafNode(self, rightNode, root, faLoc):
        """
        borrow from the right leaf node
        :param rightNode: the right leaf node
        :param root: the node currently traversed
        :param faLoc: the location in parent node
        :return: null
        """
        tempIndex = rightNode.index[0]
        tempData = rightNode.data[0]
        root.parent.index[faLoc] = rightNode.index[1]
        _, _ = self.eraseFromNode(rightNode, tempIndex, -1)
        _ = self.insertToNode(root, tempIndex, tempData)

    def borrowFromRightIndexNode(self, rightNode, root, faLoc):
        """
        borrow from the right index node
        :param rightNode: the right index node
        :param root: the node currently traversed
        :param faLoc: the location in parent node
        :return: null
        """
        tempIndex = root.parent.index[faLoc]
        tempChild = rightNode.child[0]
        root.parent.index[faLoc] = rightNode.index[0]

        # need to move the child node
        for i in range(rightNode.num - 1):
            rightNode.index[i] = rightNode.index[i + 1]
            rightNode.child[i] = rightNode.child[i + 1]
        rightNode.index[rightNode.num - 1] = 0
        rightNode.child[rightNode.num - 1] = rightNode.child[rightNode.num]
        rightNode.child[rightNode.num] = None
        rightNode.num -= 1

        root.num += 1
        root.index[root.num - 1] = tempIndex
        root.child[root.num] = tempChild
        tempChild.parent = root


    def maintainTree(self, root, faLoc):
        """
        maintain the tree
        :param root: the node currently traversed
        :param faLoc: the location in parent node
        :return: null
        """
        leftNode = None
        rightNode = None
        # print('faLoc and root.parent.num', faLoc, root.parent.num)

        # determine whether the node is on the boundary
        if faLoc > 0:
            leftNode = root.parent.child[faLoc - 1]
            # print('leftNode.num', leftNode.num)
        if faLoc < root.parent.num:
            rightNode = root.parent.child[faLoc + 1]
            # print('rightNode.num', rightNode.num)

        # determine whether the left and right node have the rest
        if leftNode != None and leftNode.num > self.minkey:
            if root.isLeaf:
                self.borrowFromLeftLeafNode(leftNode, root, faLoc)
            else:
                self.borrowFromLeftIndexNode(leftNode, root, faLoc)
        elif rightNode != None and rightNode.num > self.minkey:
            if root.isLeaf:
                self.borrowFromRightLeafNode(rightNode, root, faLoc)
            else:
                self.borrowFromRightIndexNode(rightNode, root, faLoc)
        else:

            if leftNode != None:
                # print('merge root and leftNode!')
                if root.isLeaf:
                    self.mergeLeafNode(leftNode, root, faLoc)
                else:
                    self.mergeIndexNode(leftNode, root, faLoc)
            elif rightNode != None:
                # print('merge root and rightNode!')
                if root.isLeaf:
                    self.mergeLeafNode(root, rightNode, faLoc + 1)
                else:
                    self.mergeIndexNode(root, rightNode, faLoc + 1)

    def delete(self, p, root):
        """
        delete the index p and data
        :param p: the index p to delete
        :param root: the node currently traversed
        :return: null
        """
        deleteNode, faLoc = self.eraseFromNode(root, p, -1)
        # print('deleteNode.num and self.minkey', deleteNode.num, self.minkey)
        if deleteNode.num < self.minkey:
            # print('< minkey!')
            self.maintainTree(deleteNode, faLoc)

def dfs(root):
    """
    traverse the tree
    """
    for i in range(root.num):
        print('index', root.index[i])
    if root.isLeaf:
        for i in range(root.num):
            print('data', root.data[i])
        return
    for i in range(root.num + 1):
        dfs(root.child[i])

def bfs(root):
    """
    traverse the tree
    """
    q = Queue(maxsize=1010)
    for i in range(root.num):
        print(root.index[i], end = ' ')
    print('')
    q.put(root)
    while not q.empty():
        # print('in!')
        temp = q.get()
        if temp.isLeaf:
            continue
        for i in range(temp.num + 1):
            q.put(temp.child[i])
            for j in range(temp.child[i].num):
                print(temp.child[i].index[j], end=' ')
                if temp.child[i].isLeaf:
                    print('-', temp.child[i].data[j], end=' ')
            print(end='/')
        print('')

def printLeafNode(root):
    """
    print the leaf node
    """
    while not root.isLeaf:
        root = root.child[0]
    while(root != None):
        for i in range(root.num):
            print('index and data', root.index[i], root.data[i])
        root = root.nextLeaf

if __name__ == "__main__":
    """ 
    test
    """
    a = [3, 89, 45, 62, 15, 100, 105]
    a1 = [3, 89, 45, 62, 15, 7, 26, 59, 48, 65, 32, 16, 99, 41]
    a2 = [3, 89, 45, 62, 15, 7, 26, 59, 48, 65, 32, 16, 99, 41, 100, 105]
    n = len(a)
    m = 5
    bPlusTree = BPlusTree(m)
    print('test add:')
    for i, val in enumerate(a1):
        bPlusTree.add(val, i)

    bfs(bPlusTree.root)
    # dfs(root)
    # # printLeafNode(root)
    # print('test query:')
    # print(bPlusTree.query(a2[12], bPlusTree.root))
    # print('test delete:')
    bPlusTree.delete(99, bPlusTree.root)
    # bPlusTree.delete(7, bPlusTree.root)
    # bfs(root)
    # bPlusTree.delete(105, bPlusTree.root)
    bPlusTree.delete(65, bPlusTree.root)
    print('after delete :')

    bfs(bPlusTree.root)
