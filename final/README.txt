README

CS-4641 Final Project
FIFA 19 Position Classification

There are 6 files included in this folder:
	- fifa19.csv: The uncleaned, original dataset available on kaggle
	- fifa19wr.csv: the cleaned dataset used for machine learning in the jupyter notebook
	- fifaWrangler.py: the python script used to clean the dataset
	- fifaJup.ipynb: Jupyter notebook of my project
	- fifaJup.py: the jupyter notebook downloaded as a python script that can be ran
	- report.pdf: the jupyter notebook as a .pdf
	- fifaHTML: the jupyter notebook as HTML

For running and viewing the final report I recommend using fifaJup.ipynb
If you only want to view the final report I recommend using either fifaJup.ipynb or fifaHTML

Reproducability:
	Running either fifaJup.py from terminal or 
	fifaJup.ipynb (restart and run all) should reproduce all experiments.

	(There may be slight variance in runtime and performance because of random splits
	but the overall conclusions will still hold)

Installing Dependencies:
	pandas: pip install pandas
	numpy: pip install numpy
	seaborn: pip install seaborn
	scikit-learn: pip install scikit-learn
	scipy: pip install scipy
	pytorch: pip3 install torch==1.3.1+cpu torchvision==0.4.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
