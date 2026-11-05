src = open('cfb_gap13.py').read()
exec(src.split('\nrep("v3", v3)\nfor nm')[0])
fit.to_parquet("cfb_fit.parquet"); te.to_parquet("cfb_te_all26.parquet"); print(len(fit), len(te))
