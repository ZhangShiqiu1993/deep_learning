class Node(object):
	def __init__(self, inbound_nodes=[]):
		self.inbound_nodes = inbound_nodes
		self.outbound_nodes = []

		for inbound_node in self.inbound_nodes:
			inbound_node.outbound_nodes.append(self)

		self.value = None

	def forward(self):
		"""
		Forware propagation
		
		Compute the output value based on `ibound_nodes`
		and store the result in self.value
		:return: 
		"""
		raise NotImplemented


class Input(Node):
	def __init__(self):
		Node.__init__(self)

	def forward(self, value=None):
		if value is not None:
			self.value = value


class Add(Node):
	def __init__(self, x, y):
		Node.__init__(self, [x, y])

	def forward(self):
		pass
