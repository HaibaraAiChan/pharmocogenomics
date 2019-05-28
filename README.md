# Pharmocogenomics  (MLP)   
## DrugBank dataset (8,000)  

* **Drug Structure & Protein Sequence**

	* **training dataset** 
		**training-dataset**
		```
		pubchem-id:  drug A
		ensemble-id:  protein B
		confidence score:  interaction score of A & B 
		Drug-Bank-id: drug A
		uniport-id: protein B
		mol2vec-000 ~ 299: chemical structure 300 feature of drug A
		0_y ~ 99: sequence 100 feature of protein B (combine 3 rows which means 3 different sequence split methods to generate 300 length vector)
		class: 0 or 1 (negative or positive)
		```
		
	











## intersection DrugBank & STITCH dataset (271)
* **Drug Structure & Protein Sequence**

	* **training dataset** 

		* **including:**  
			1. drug structure vector (**300**)   
			2. protein sequence vector (**300**)  


		**training-dataset-rf-mlp.csv**
		```
		pubchem-id:  drug A
		ensemble-id:  protein B
		confidence score:  interaction score of A & B 
		Drug-Bank-id: drug A
		uniport-id: protein B
		mol2vec-000 ~ 299: chemical structure 300 feature of drug A
		0_y ~ 99: sequence 100 feature of protein B (combine 3 rows which means 3 different sequence split methods to generate 300 length vector)
		class: depends on confidence score 
			score 	 <300: class 3
			score 300~500: class 2
			score 	 >500: calss 1
		```
	


	* **test dataset**   
		- [ ]  drug structure vector(300) & protein sequence vector(300)  
	
	* **MLP model:**  
		**input:**  drug structure vector(300) & protein sequence vector(300)     
		**output:**  interaction score class (1,2,3)  
