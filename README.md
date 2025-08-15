# Tesi-scvi
Pancreas data folder: https://figshare.com/ndownloader/files/24539828
Scrittura tesi circa 30 pagine 
inizio di agosto fine testing
[ ] Early stopping (implementarlo sulle loss test, prendendo il 25esimo percentile essendo che farlo sul train diventa complicato aggregrare)
non avendo early stopping mettiamo come num di epoche 100 per confrontare, per far vedere in modo chiaro l'andamento della loss alla modifica del numero di epoche
[ ] Calcolare varianza locale per bacth e calcolare media pesata delle varianze (messaggio teams)
[-] 2000 geni più varianti federato (NON è uguale, non ho incluso i batch come fa la funzione sc.pp.highly_variable_genes(pancreas_ref, n_top_genes=2000, batch_key="tech"))

[ ] provare anche 1 client per ogni tech, e valutare i cluster
[x] valutare con metriche la clusterizzazione -> tabella
[x] riprodurre plot sul test (https://genomebiology.biomedcentral.com/articles/10.1186/s13059-025-03684-6) figura 3a. confronto tra solo umap. 
[x] controllare come il modello per client come riconosce il test
[x] sistemare il test, tenendo fuori solo le due tecnologie
[x] Cambiando parametri federati come cambia la loss del test, Testare diversi parametri federati?

[ ] aggiungere tabella performance sul grafico della loss
[ ] aggiungere trattini per clients
elenco plot risultati: 
    [x] loss test (anche train ma non so) (flower manca ealry stopping integrato)
    [x] slide 1 Federati con diversi parametri vs centralizzato
    [x] slide 2 centralizzato vs best fed vs clients
    [x] slide 3 Umap come figura 3a (forse anche in due slides)
    
elenco argomenti teorici:
    [ ] Federated Learning
    [ ] Single Cell analysis
    [ ] SCVI
    [ ] VAE

FL + SingleCell = SCVI
VAE -> SCVI



