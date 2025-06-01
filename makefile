.PHONY:	run

# Get the conda base installation directory.
CONDA_BASE := $(shell conda info --base)

install:
				@echo "Running installation script..."
				@bash src/install.sh

run:
				@echo "Activating conda environment 'cenv' and running clustering analysis..."
				@source $(CONDA_BASE)/etc/profile.d/conda.sh && conda activate cenv && python src/node2vec_clustering_analysis.py

notebook:
				@echo "Activating conda environment 'cenv' and launching Jupyter Notebook..."
				@bash -c "source $(CONDA_BASE)/etc/profile.d/conda.sh && conda activate cenv && jupyter notebook jp/Node2Vec_clustering_analysis.ipynb"
