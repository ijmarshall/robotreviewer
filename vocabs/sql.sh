# mysql -uroot umls -e "SELECT cui, str, sab FROM MRCONSO where suppress='N' and (sab='MSH' or sab='MDR' or sab='SNOMEDCT_US' or sab='ATC' or sab='RXNORM' or sab='ICD10') and stt='PF';" > cui_str.csv
# mysql -uroot umls -e "SELECT cui, str, sab FROM MRCONSO where suppress='N';" > umls_full_index.csv
mysql -uroot umls -e "SELECT cui1, cui2 FROM MRREL WHERE sab IN ('MSH', 'SNOMEDCT_US', 'RXNORM', 'ATC', 'MDR', 'ICD10') AND rel='PAR';" > cui_graph.csv
