from nvcr.io/nvidia/pytorch:22.01-py3
RUN pip install --upgrade pip
RUN pip install -U jupyterlab-widgets
RUN pip install -U ipywidgets
RUN pip install termcolor
RUN pip install wandb transformers
RUN pip install plotly==5.13.1
RUN pip install seaborn
#RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager jupyterlab-plotly
ENTRYPOINT ["jupyter", "lab", "--port=8888", "--no-browser", "--allow-root", "--ip='*'"]
