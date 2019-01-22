# class definition for a tree made of treeNodes

Class treeGraph():
	"""
	treeGraph is an object to contain a collection of nodes.
	
	Methods:
		addNode
		toString

	Attributes:
		nodes - a list of node objects
		nNodes - the length of 'nodes'
	"""
	__init__(self):
		self.nNodes = 0
		self.nodes

	def addNode(self, node):
		"""
		Adds a node to the tree.

		Parameters:
			node - the node to add to the tree
		"""
		self.nodes.append(node)

	def findLeaves(self):
		"""
		Identifies the leaves in the 

	def toString(self):
		"""
		Converts the structure of the tree to an ASCII readable format for printing
		"""
		pass # TODO
