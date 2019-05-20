#description     Generate solution
#author			 Alex Tselikov < atselikov@gmail.com >
#==============================================================================
mkdir datasets
mkdir subs 
mkdir level2
python preprocess.py
python level1run.py
python level2blend.py