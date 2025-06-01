.PHONY:	run

run:
				@echo "Activating virtual environment and running clustering analysis..."
				@source venv/bin/activate && python src/node2vec_clustering_analysis.py
