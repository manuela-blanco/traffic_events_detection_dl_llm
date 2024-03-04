import subprocess
# transitorj
accounts = ['OperacoesRio','Arteris_AFL','_ecoponte','LinhaAmarelaRJ','CETSP_',
           '_dersp']

for account in accounts:
    subprocess.run(["py", "extraction_script.py", "--twitter_account", account])