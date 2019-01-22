# class definition for the nodes of a tree

Class treeNode():
	"""
	treeNode is an object that contains the information at each split of a tree.

	Attributes:
		iD - the node's ID number
		parent - the ID number of the parent's node
		leftChild - the left child of the node
		rightChild - the right child of the node
		impurity - the impurity of the node's contents
		feature - the feature that is contained in the node, if it is a split
		threshold - the threshold used to split on the feature
		nSamples - the number of samples in the node (or that pass through it)
	"""
	__init__(self, iD=0, parent=None, leftChild=None, rightChild=None, impurity=None, feature='', threshold=None, nSamples=None, isVisited=False):
		self.iD = iD
		self.parent = parent
		self.leftChild = leftChild
		self.rightChild = rightChild
		self.impurity = impurity
		self.feature = feature
		self.threshold = threshold
		self.nSamples = nSamples

