rm -rf ./output
mkdir ./output
export profile="wiki10k" && python main.py > ./output/wiki10k.log
export profile="twitter" && python main.py > ./output/twitter.log
export profile="ppi" && python main.py > ./output/ppi.log
export profile="dblp" && python main.py > ./output/dblp.log
export profile="blogcatalog" && python main.py > ./output/blogcatalog.log