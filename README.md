# Pharmocogenomics  (MLP)
* **Drug Structure & Protein Sequence**

	**training dataset description**  

	**including:**  
	* drug structure vector (**300**)   
	* protein sequence vector (**300**)  


	**training-dataset-rf-mlp.csv**
	```
	pubchem-id:  drug A
	ensemble-id:  protein B
	confidence score:  interaction score of A & B
	Drug-Bank-id: drug A
	uniport-id: protein B
	mol2vec-000 ~ 299: chemical structure 300 feature of drug A
	0_y ~ 99: sequence 100 feature of protein B (combine 3 rows which means 3 different sequence split methods to generate 300 length vector)
	```
