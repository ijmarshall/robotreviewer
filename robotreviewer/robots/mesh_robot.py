'''
The MeshBot consumes text snippets and extracts MESH terms.
'''

from robotreviewer.textprocessing import minimap

class MeshRobot():
	"""
	minimap wrapper for API
	"""
	def api_annotate(self, articles):		
		return [minimap.get_unique_terms([article['snippet']]) for article in articles]

