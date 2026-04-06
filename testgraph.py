from knowledge_graph import KnowledgeGraph
kg = KnowledgeGraph.load_from_db("./movex_kg.db")
print(kg.get_products_by_series("880"))
print(kg.get_products_by_series(880))
print(kg.get_products_by_series("series:880"))