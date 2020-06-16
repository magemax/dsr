import pandas as pd
import numpy as np


def xlsxtocsv(filename):
    col_name_separator="#" #les noms de colonnes viennent sur 2 lignes. On les concatène ainsi
    print("transforming")
    df=pd.read_excel(filename,header=[0,1],dtype=str)
    df.columns = df.columns.map(col_name_separator.join).str.strip(col_name_separator)
    nom_fichier=filename.split(".")[0]
    print("read. Now saving with better format")
    df.to_csv(nom_fichier+".csv", index=False)
    df.to_pickle(nom_fichier+".pkl")
    print("transformed all")

annee_calcul=2019
annee_precedente=annee_calcul-1
annee_encoreavant=annee_calcul-2 #le format étant pas le même pour 2017, on est dans la merde

#Fichiers obtenus en lançant la fonction xlsxtocsv sur les fichiers YYYY_communes.xlsx publiés
# par la DGCL.
lastdata= pd.read_pickle("{}_communes.pkl".format(annee_calcul))
beforedata=pd.read_pickle("{}_communes.pkl".format(annee_precedente))

def searchfield(tofind, table=lastdata):
    return [k for k in table.columns if tofind.lower() in k.lower()]

def printsearchfield(tofind):
    src=searchfield(tofind)
    for i in src:
        print(i)
    if not src:
        print("no field contains", tofind)

def countgroupby(fieldname, whom=lastdata):
    return whom.groupby([fieldname])[[whom.columns[0]]].count()


# les variables string décrivent des colonnes (existantes ou créées) dans la table de données unique qui contient les
# données sur les colonnes
# les colonnes dont le nom commence par actual_ représentent des champs calculés, en général importés poupr
# comparaison ou parce qu'on ne dispose pas des moyens de tout bien calculer (exemple : les garanties)

code_comm ="Informations générales#Code INSEE de la commune"
is_outre_mer = "commune d'outre mer"
departement="Informations générales#Code département de la commune"
lastdata[is_outre_mer] = (lastdata[departement].str.len()>2)  & (~lastdata[departement].str.contains("A")) & (~lastdata[departement].str.contains("B"))
#lastdata=lastdata[(lastdata[code_comm] == "57163") | (lastdata[code_comm] == "87116")]

# Exemple d'usage des fonctions printsearchfield (utilisé pour rechercher des champs facilement)
# et countgroupby (utilisé pour obtenir un comptage d'un certain champ
printsearchfield("outre")
print(countgroupby(is_outre_mer))

# CALCUL DE LA DSR

# Vrais montants
# bc = fraction bourg-centre
# pq = fraction péréquation
# cible = fraction cible
actual_montant_elig_bc = "Dotation de solidarité rurale Bourg-centre#Montant de la commune éligible"
actual_montant_dsr_bc = "Dotation de solidarité rurale Bourg-centre#Montant global réparti"
actual_montant_dsr_pq = "Dotation de solidarité rurale - Péréquation#Montant global réparti (après garantie CN)"
actual_montant_dsr_cible = "Dotation de solidarité rurale - Cible#Montant global réparti"
actual_elig_bc = "eligible bc"
lastdata[actual_elig_bc] = (lastdata[actual_montant_elig_bc]>0)

# 1. Fraction bourg-centre

#variables
dsr_bc_seuil_pop_1 = 10_000
dsr_bc_seuil_pop_2 = 20_000
seuil_part_canton = 0.15
tolerance = 0.000001
dsr_bc_pop_max_agglo = 250_000
dsr_bc_part_max_pop_departement = 0.1
dsr_bc_taille_max_chef_lieu_canton = 10_000
dsr_bc_ratio_max_pot_fin = 2
dsr_bc_taille_max_plus_grande_commune_agglo = 100_000


dsr_bc="Dotation de solidarité rurale Bourg-centre#Montant global réparti"
# 1.1 - < 10000 hab
pop_dgf="Informations générales#Population DGF Année N'"
pop_plaf="Dotation de solidarité rurale Bourg-centre#Population DGF plafonnée"
pop_insee="Informations générales#Population INSEE Année N "
part_canton="Dotation de solidarité rurale Bourg-centre#Pourcentage de la population communale dans le canton"
bur_centr="Dotation de solidarité rurale Bourg-centre#Bureaux centralisateurs"
chef_lieu_canton="Dotation de solidarité rurale Bourg-centre#Code commune chef-lieu de canton au 1er janvier 2014"
is_chef_lieu_canton = "is_chef_lieu_canton"

chef_lieu_canton_corrected = "chef-lieu canton corrigé"
lastdata[chef_lieu_canton_corrected] = lastdata[chef_lieu_canton].apply(lambda x: str(x).zfill(5))
lastdata[is_chef_lieu_canton]= (lastdata[chef_lieu_canton_corrected]==lastdata[code_comm])
# According to stack, this would also work :
# lastdata[is_chef_lieu_canton]= lastdata[code_comm].str.lstrip('0') == lastdata[chef_lieu_canton].astype(str)
# or this :
# lastdata[is_chef_lieu_canton]= [x.endswith(str(y)) for x, y in lastdata[[code_comm,chef_lieu_canton]].values]
elig_bc10000= "eligible_bourg_centre type 1" #pour les communes de moins de 10000 hab
print(countgroupby(is_chef_lieu_canton,lastdata))
#Data is not always sufficient to discriminate (when rounded percentage is very close to threshold)
#so we recompute a better estimate in these cases, but it's only used when we are close to the threshold
#Cause we somehow get weird results...
somme_pop_par_canton = lastdata.groupby(chef_lieu_canton_corrected)[[pop_dgf]].sum()
pop_dgf_canton = "Population DGF totale du canton"
somme_pop_par_canton.columns=[pop_dgf_canton]
lastdata=lastdata.merge(somme_pop_par_canton,left_on=chef_lieu_canton_corrected,right_index=True, suffixes=["","_cheflieucanton"])
part_pop_canton_corrected = part_canton+" Corrigée"
lastdata[part_pop_canton_corrected]=lastdata[pop_dgf]/lastdata[pop_dgf_canton]
above_seuil = "is above 15% population of canton"
above_seuil_source = "is equal to 15% of canton according to original data"
lastdata[above_seuil_source] = (lastdata[part_canton] > seuil_part_canton)
lastdata[above_seuil] =  ((lastdata[part_canton]>seuil_part_canton + tolerance) | (((lastdata[part_canton] - seuil_part_canton).abs() <= tolerance) & (lastdata[part_pop_canton_corrected] >= seuil_part_canton)))

print(lastdata.groupby([above_seuil,above_seuil_source])[code_comm].count())
lastdata[elig_bc10000]= (lastdata[is_chef_lieu_canton] | (lastdata[bur_centr]=="OUI") | lastdata[above_seuil])& (lastdata[pop_plaf]<dsr_bc_seuil_pop_1) & (~lastdata[is_outre_mer])
print(countgroupby(elig_bc10000,lastdata))

#Exception 1.1.1.a
#retrait des unités urbaines trop grosses
pop_agglo="Dotation de solidarité rurale Bourg-centre#Population DGF des communes de l'agglomération"
pop_departement="Dotation de solidarité rurale Bourg-centre#Population départementale de référence de l'agglomération"
lastdata.loc[(lastdata[pop_agglo]>dsr_bc_pop_max_agglo) | (lastdata[pop_agglo]>lastdata[pop_departement]*dsr_bc_part_max_pop_departement), elig_bc10000] = False
print(countgroupby(elig_bc10000,lastdata))

#Exception 1.1.1.b : info non présente dans le fichier ?? (plus grosse commune > 100000 hab... on peut espérer que y a sonneper)
chef_lieu_departement_dans_agglo = "Dotation de solidarité rurale Bourg-centre#Chef-lieu de département agglo"
lastdata.loc[(lastdata[chef_lieu_departement_dans_agglo]=="OUI"), elig_bc10000] = False
print(countgroupby(elig_bc10000,lastdata))
max_pop_plus_grosse_commune = lastdata.groupby(pop_agglo)[[pop_dgf]].max()
somme_pop_plus_grosse_commune =  lastdata.groupby(pop_agglo)[[pop_dgf]].sum()
print(somme_pop_plus_grosse_commune[somme_pop_plus_grosse_commune.index!=somme_pop_plus_grosse_commune[somme_pop_plus_grosse_commune.columns[0]]])
nom_plus_grosse_commune = "Population plus grande commune de l'agglomération"
max_pop_plus_grosse_commune.columns = [nom_plus_grosse_commune]
print(len(lastdata))
lastdata=lastdata.merge(max_pop_plus_grosse_commune,left_on=pop_agglo,right_index=True).sort_index()
print(len(lastdata))
lastdata.loc[(lastdata[pop_agglo]>0) & (lastdata[nom_plus_grosse_commune]>dsr_bc_taille_max_plus_grande_commune_agglo), elig_bc10000] = False
print(countgroupby(elig_bc10000,lastdata))

# Exception 1.1.2
# retrait canton dont chef lieu > 10000 hab

table_chef_lieu_canton=lastdata[[code_comm,pop_dgf,bur_centr]]
pop_dgf_chef_lieu_canton= pop_dgf+" du chef-lieu de canton"
chef_lieu_canton_bur_centr = bur_centr + " pour le chef-lieu de canton"
table_chef_lieu_canton.columns=[code_comm, pop_dgf_chef_lieu_canton, chef_lieu_canton_bur_centr]
lastdata=lastdata.merge(table_chef_lieu_canton,left_on=chef_lieu_canton_corrected,right_on=code_comm,how="left", suffixes=("","_Chef_lieu"))

lastdata.loc[(lastdata[pop_dgf_chef_lieu_canton]>dsr_bc_taille_max_chef_lieu_canton) & (lastdata[bur_centr]=="NON"), elig_bc10000] = False
print(countgroupby(elig_bc10000,lastdata))

# Exception 1.1.3
# exclusion des communes dont le potentiel financier est supérieur au double de la moyenne
pot_fin_10000 = "Dotation de solidarité rurale Bourg-centre#Potentiel financier -10 000"
pot_fin_par_hab ="Potentiel fiscal et financier des communes#Potentiel financier par habitant"
lastdata.loc[(lastdata[pot_fin_par_hab]>dsr_bc_ratio_max_pot_fin*lastdata[pot_fin_10000]), elig_bc10000] = False

print(countgroupby(elig_bc10000,lastdata))

# 1.2 : 10000 <= pop < 20000 pour les chefs lieux d'arrondissement

is_chef_lieu_arrondissement = "Dotation de solidarité rurale Bourg-centre#Chef-lieu d'arrondissement au 31 décembre 2014"
elig_bc20000= "eligible_bourg_centre type 2" #pour les communes de plus de 10000 et moins de 20000 hab

lastdata[elig_bc20000]= (lastdata[is_chef_lieu_arrondissement]=="OUI") & (lastdata[pop_plaf]>=10000) &  (lastdata[pop_plaf]<20000) & (~lastdata[is_outre_mer])
print(countgroupby(elig_bc20000,lastdata))
# Exclusion 1.2.1.a
#retrait des unités urbaines trop grosses
lastdata.loc[(lastdata[pop_agglo]>dsr_bc_pop_max_agglo) | (lastdata[pop_agglo]>lastdata[pop_departement]*dsr_bc_part_max_pop_departement), elig_bc20000] = False
print(countgroupby(elig_bc20000,lastdata))

#Exception 1.2.1.b : info non présente dans le fichier ?? (plus grosse commune > 100000 hab... on peut espérer que y a sonneper)
lastdata.loc[(lastdata[chef_lieu_departement_dans_agglo]=="OUI"), elig_bc20000] = False
print(countgroupby(elig_bc20000,lastdata))
lastdata.loc[(lastdata[pop_agglo]>0) & (lastdata[nom_plus_grosse_commune]>dsr_bc_taille_max_plus_grande_commune_agglo), elig_bc20000] = False
print(countgroupby(elig_bc20000,lastdata))

# Exception 1.2.3
# exclusion des communes dont le potentiel financier est supérieur au double de la moyenne
pot_fin_10000 = "Dotation de solidarité rurale Bourg-centre#Potentiel financier -10 000"
pot_fin_par_hab ="Potentiel fiscal et financier des communes#Potentiel financier par habitant"
lastdata.loc[(lastdata[pot_fin_par_hab]>dsr_bc_ratio_max_pot_fin*lastdata[pot_fin_10000]), elig_bc20000] = False

print(countgroupby(elig_bc20000,lastdata))

elig_bc = "Eligible bourg centre"
lastdata[elig_bc] = lastdata[elig_bc20000] | lastdata[elig_bc10000]

#Fraction bourg-centre : montant de garanti
suffix_last_year = " Année précédente"
lastdata=lastdata.merge(beforedata[[code_comm,actual_montant_elig_bc,actual_montant_dsr_pq,actual_montant_dsr_cible]],on=code_comm,suffixes=["",suffix_last_year],how="left")

montant_previous_year_bc = "Dotation de solidarité rurale Bourg-centre#Montant de la commune éligible"+suffix_last_year
montant_previous_year_pq = actual_montant_dsr_pq+suffix_last_year
montant_previous_year_cible = actual_montant_dsr_cible + suffix_last_year
non_elig_bc_garantie = "Montant garanti, DSR - fracion bourg-centre"
lastdata[non_elig_bc_garantie]= ((~lastdata[elig_bc]) * (lastdata[montant_previous_year_bc]) * 0.5 + 0.5).astype(int)
actual_montant_garanti_bc = "Dotation de solidarité rurale Bourg-centre#Montant de la garantie de sortie"
#lastdata["diff"] = lastdata[actual_montant_garanti_bc] - lastdata[non_elig_bc_garantie]
#lastdata.to_csv("mailin2.csv")

#Not implementing here :
# La garantie que 2018 == 2017 pour les communes sorties en 2017
# So a few discrepancies w.r.t. the file...
#thus, we use actual_montant_garanti_bc   and not non_elig_bc_garantie

montant_garantie_dsr_bourg_centre = actual_montant_garanti_bc

# Repartition montant
cap_pop_pour_montant_score= 10000
cap_effort_fiscal = 1.2
value_mult_zrr= 1.3

total_budget_dsr_fraction_bc = 545248129  #D'où ça vient???
total_to_attribute_bc = total_budget_dsr_fraction_bc - lastdata[montant_garantie_dsr_bourg_centre].sum()

score_dsr_bc = "Score commune eligible fraction bourg centre"
facteur_pfi = "facteur potentiel financier DSR bourg-centre"
lastdata[facteur_pfi] = 2 - lastdata[pot_fin_par_hab] / lastdata[pot_fin_10000]
effort_fiscal = "Effort fiscal#Effort fiscal"
facteur_zrr = "facteur ZRR DSR bourg centre"
is_zrr = "Dotation de solidarité rurale - Bourg-centre#Commune située en ZRR"
lastdata[facteur_zrr] = 1 + (value_mult_zrr-1) * (lastdata[is_zrr]=="OUI")
lastdata[score_dsr_bc] = (lastdata[elig_bc]
                          * np.minimum(cap_pop_pour_montant_score, lastdata[pop_plaf])
                          * np.minimum(lastdata[effort_fiscal],cap_effort_fiscal)
                          * lastdata[facteur_pfi]
                          * lastdata[facteur_zrr])
total_score_bc = lastdata[score_dsr_bc].sum()
print(total_score_bc, total_to_attribute_bc, total_to_attribute_bc/total_score_bc)
valeur_point_bc=total_to_attribute_bc/total_score_bc
print("valeur point", valeur_point_bc)

dsr_bc_montant_eligible = "DSR bourg centre éligible Montant attribué communes éligibles"

""" Ce bout de code effectue une recherche dichotomique pour déterminer la valeur du point qui dépense pile le budget...
Il semble que ce soit pas comme ça que marche la loi..

valeur_point_max=2*valeur_point_bc
valeur_point_bc=0
precision_vp=0.0000000001
nbiterations=0

while valeur_point_max>precision_vp:
    valeur_point_bc+=valeur_point_max
    lastdata[dsr_bc_montant_eligible]=valeur_point_bc*lastdata[score_dsr_bc]
    print(lastdata[dsr_bc_montant_eligible].sum())
    lastdata.loc[(lastdata[montant_previous_year_bc]>0) & (lastdata[elig_bc]), dsr_bc_montant_eligible] = np.maximum(0.9 * lastdata[montant_previous_year_bc], lastdata[dsr_bc_montant_eligible])
    #lastdata[dsr_bc_montant_eligible]= np.maximum(0.9 * lastdata[montant_previous_year_bc], lastdata[dsr_bc_montant_eligible])
    print(lastdata[dsr_bc_montant_eligible].sum())
    print(len(lastdata[lastdata[score_dsr_bc]>0]))
    lastdata.loc[(lastdata[montant_previous_year_bc]>0) & (lastdata[elig_bc]), dsr_bc_montant_eligible] = np.minimum(1.2* lastdata[montant_previous_year_bc], lastdata[dsr_bc_montant_eligible])
    print(len(lastdata[lastdata[score_dsr_bc]>0]))
    print(lastdata[dsr_bc_montant_eligible].sum())
    if lastdata[dsr_bc_montant_eligible].sum()>total_to_attribute_bc:
        valeur_point_bc-=valeur_point_max
    valeur_point_max/=2
    nbiterations+=1
print("real vp", valeur_point_bc)
print("nb iterations", nbiterations)

lastdata.to_csv("jipopt.csv")"""

#valeur_point_bc=37.37858154
lastdata[dsr_bc_montant_eligible] = valeur_point_bc * lastdata[score_dsr_bc]
print(lastdata[dsr_bc_montant_eligible].sum())
#Montant garantie au niveau précédent pour les communes éligibles
lastdata.loc[(lastdata[montant_previous_year_bc] > 0) & (lastdata[elig_bc]), dsr_bc_montant_eligible] = np.maximum(
    0.9 * lastdata[montant_previous_year_bc], lastdata[dsr_bc_montant_eligible])
print(lastdata[dsr_bc_montant_eligible].sum())
lastdata.loc[(lastdata[montant_previous_year_bc] > 0) & (lastdata[elig_bc]), dsr_bc_montant_eligible] = np.minimum(
    1.2 * lastdata[montant_previous_year_bc], lastdata[dsr_bc_montant_eligible])
print(lastdata[dsr_bc_montant_eligible].sum())
# if lastdata[dsr_bc_montant_eligible].sum() > total_to_attribute_bc:
#     valeur_point_bc -= valeur_point_max

dsr_bc_montant_total = "DSR bourg centre Montant attribué total"

lastdata[dsr_bc_montant_total] = lastdata[dsr_bc_montant_eligible] + lastdata[montant_garantie_dsr_bourg_centre]

#Partie 2 : Fraction péréquation

nb_hab_pq = 10000
elig_pq= "Eligible à la fraction de péréquation"
pot_fin_strate = "Potentiel fiscal et financier des communes#Potentiel financier moyen de la strate"
lastdata[elig_pq]=(lastdata[pop_dgf]<nb_hab_pq) & (lastdata[pot_fin_par_hab]<2*lastdata[pot_fin_strate]) & (~lastdata[is_outre_mer])

actual_montant_pq_1 ="Dotation de solidarité rurale - Péréquation#Part Pfi (avant garantie CN)"
actual_elig_pq = "Eligible péréquation d'après la DGCL"
lastdata[actual_elig_pq] = (lastdata[actual_montant_pq_1]>0)
# Montant

montant_pq = "DSR fraction péréquation Montant attribué éligible"
montant_pq_1 = montant_pq + " - part potentiel financier par habitant"
montant_pq_2 = montant_pq + " - part voirie"
montant_pq_3 = montant_pq + " - part enfants"
montant_pq_4 = montant_pq + " - part potentiel financier par hectare"

total_budget_dsr_fraction_pq = 645_050_872
total_garanties_pq = 7_403_713 # ????? Pas dans la loi ni dans la note DGCL
total_to_attribute_pq = total_budget_dsr_fraction_pq - total_garanties_pq

# part 2.1 : potentiel financier
weight_dsr_pq_1 = 0.3
score_dsr_pq_1 = montant_pq_1.replace("Montant attribué", "score")
facteur_pot_fin_strate = "facteur potentiel financier DSR strate"
lastdata[facteur_pot_fin_strate] = 2 - lastdata[pot_fin_par_hab] / lastdata[pot_fin_strate]

lastdata[score_dsr_pq_1] = (lastdata[elig_pq]
                          * lastdata[pop_dgf]
                          * np.minimum(lastdata[effort_fiscal], cap_effort_fiscal)
                          * lastdata[facteur_pot_fin_strate])
print("to attr", weight_dsr_pq_1*total_to_attribute_pq, "tot score", lastdata[score_dsr_pq_1].sum(), "VP :", weight_dsr_pq_1*total_to_attribute_pq / lastdata[score_dsr_pq_1].sum())
valeur_point_pq_1 = weight_dsr_pq_1*total_to_attribute_pq / lastdata[score_dsr_pq_1].sum()
lastdata[montant_pq_1] = lastdata[score_dsr_pq_1] * valeur_point_pq_1

# part 2.2 : voirie
weight_dsr_pq_2 = 0.3
score_dsr_pq_2 = montant_pq_2.replace("Montant attribué", "score")
longueur_voirie ="Dotation de solidarité rurale - Péréquation#Longueur de voirie en mètres"
is_montagne = "Dotation de solidarité rurale - Péréquation#Commune située en zone de montagne"
is_insulaire = "Dotation de solidarité rurale - Péréquation#Commune insulaire"
lastdata[score_dsr_pq_2] = lastdata[elig_pq] * lastdata[longueur_voirie] * (1 + ((lastdata[is_montagne]=="OUI") | (lastdata[is_insulaire]=="OUI")))
print("to attr", weight_dsr_pq_2*total_to_attribute_pq, "tot score", lastdata[score_dsr_pq_2].sum(), "VP :", weight_dsr_pq_2*total_to_attribute_pq / lastdata[score_dsr_pq_2].sum())
valeur_point_pq_2 = weight_dsr_pq_2*total_to_attribute_pq / lastdata[score_dsr_pq_2].sum()
lastdata[montant_pq_2] = lastdata[score_dsr_pq_2] * valeur_point_pq_2

# part 2.3 : enfants
weight_dsr_pq_3 = 0.3
score_dsr_pq_3 = montant_pq_3.replace("Montant attribué", "score")
nb_hab_3_16 = "Dotation de solidarité rurale - Péréquation#Population 3 à 16 ans"
lastdata[score_dsr_pq_3] = lastdata[elig_pq] * lastdata[nb_hab_3_16]
print("to attr", weight_dsr_pq_3*total_to_attribute_pq, "tot score", lastdata[score_dsr_pq_3].sum(), "VP :", weight_dsr_pq_3*total_to_attribute_pq / lastdata[score_dsr_pq_3].sum())
valeur_point_pq_3 = weight_dsr_pq_3*total_to_attribute_pq / lastdata[score_dsr_pq_3].sum()
lastdata[montant_pq_3] = lastdata[score_dsr_pq_3] * valeur_point_pq_3

# part 2.4 : potentiel financier par hectare
weight_dsr_pq_4 = 0.1
score_dsr_pq_4 = montant_pq_4.replace("Montant attribué", "score")
pot_fin_10000_hectare= "Dotation de solidarité rurale - Péréquation#Potentiel financier par hectare - 10 000"
pot_fin_par_hectare =  "Potentiel fiscal et financier des communes#Potentiel financier superficiaire"
lastdata[score_dsr_pq_4] = lastdata[elig_pq] * lastdata[pop_dgf] * np.maximum(0, 2 - lastdata[pot_fin_par_hectare] / lastdata[pot_fin_10000_hectare])
print("to attr", weight_dsr_pq_4*total_to_attribute_pq, "tot score", lastdata[score_dsr_pq_4].sum(), "VP :", weight_dsr_pq_4*total_to_attribute_pq / lastdata[score_dsr_pq_4].sum())
valeur_point_pq_4 = weight_dsr_pq_4*total_to_attribute_pq / lastdata[score_dsr_pq_4].sum()
lastdata[montant_pq_4] = lastdata[score_dsr_pq_4] * valeur_point_pq_4

lastdata[montant_pq] = lastdata[montant_pq_1] + lastdata[montant_pq_2] +lastdata[montant_pq_3] + lastdata[montant_pq_4]
print(lastdata[montant_pq].sum())
lastdata.loc[(lastdata[montant_previous_year_pq] > 0) & (lastdata[elig_pq]), montant_pq] = np.maximum(
    0.9 * lastdata[montant_previous_year_pq], lastdata[montant_pq])
print(lastdata[montant_pq].sum())
lastdata.loc[(lastdata[montant_previous_year_pq] > 0) & (lastdata[elig_pq]), montant_pq] = np.minimum(
    1.2 * lastdata[montant_previous_year_pq], lastdata[montant_pq])
print(lastdata[montant_pq].sum())


dsr_pq_montant_total = "DSR fraction péréquation Montant attribué total"

lastdata[dsr_pq_montant_total] = lastdata[montant_pq]   #pour l'instant, pas de moyen de calculer la garantie ni de la trouver dans le fichier

#DSR partie 3 : part cible

#Eligibilité : on classe les communes éligibles à une des 2 premieres fractions de l'ISR
# selon un indicateur ad hoc, et on sélectionne les 10000 premieres

# Bon déjà on n'a pas tout le matos pour calculer l'indicateur...
#Il nous manque le revenu moyen par catégorie, aussi on le recalcule nous même...
#On pourrait aussi le recalculer nous même à partir des scores synthétiques présents.

strate = "Informations générales#Strate démographique Année N"
revenu_total_commune="Dotation de solidarité urbaine#Revenu imposable des habitants de la commune"
actual_indice_synthetique_cible= "Dotation de solidarité rurale - Cible#Indice synthétique"

tableau_donnees_par_strate= lastdata[(~lastdata[is_outre_mer])].groupby(strate)[[pop_insee,revenu_total_commune]].sum()
revenu_moyen_strate= " Revenu imposable moyen par habitant de la strate"
tableau_donnees_par_strate[revenu_moyen_strate] = tableau_donnees_par_strate[revenu_total_commune]/tableau_donnees_par_strate[pop_insee]
nb_communes_cible = 10000
poids_pot_fin_eligibilite_cible = 0.7
poids_revenu_eligibilite_cible = 0.3

print(tableau_donnees_par_strate)

lastdata=lastdata.merge(tableau_donnees_par_strate[[revenu_moyen_strate]],left_on=strate,right_index=True)
print(len(lastdata))
revenu_moyen_commune = "Dotation de solidarité urbaine#Revenu imposable par habitant"
# Certains revenus moyens sont missing...
# On essaye de les remplir grâce à notre super equation:
# RM = (0.3*RMStrate)/(IS-0.7 * PF(strate)/PF)
revenu_moyen_commune_corrige= revenu_moyen_commune+" corrigé"
lastdata[revenu_moyen_commune_corrige]=lastdata[revenu_moyen_commune]
lastdata.loc[(lastdata[revenu_moyen_commune_corrige]==0) & (lastdata[pop_insee]>0), revenu_moyen_commune_corrige]= (
    poids_revenu_eligibilite_cible * lastdata[revenu_moyen_strate] / (lastdata[actual_indice_synthetique_cible] - poids_pot_fin_eligibilite_cible * lastdata[pot_fin_strate]/lastdata[pot_fin_par_hab])
)
score_eligibilite_dsr_cible = "Score dotation pour éligibilité part-cible"


lastdata[score_eligibilite_dsr_cible] = ((lastdata[elig_bc] | lastdata[elig_pq]) & (lastdata[pop_dgf]<10000)) * (
    poids_pot_fin_eligibilite_cible
    * lastdata[pot_fin_strate]/lastdata[pot_fin_par_hab]
    + poids_revenu_eligibilite_cible
    * lastdata[revenu_moyen_strate] /lastdata[revenu_moyen_commune_corrige]
)
# Cas des communes sans habitants : on met à 0.3 d'indice synthétique
lastdata.loc[lastdata[pop_dgf]==0, score_eligibilite_dsr_cible]= 0.3

rang_indice_synthetique_cible= " DSR rang pour eligibilité part-cible"

lastdata[rang_indice_synthetique_cible] = lastdata[score_eligibilite_dsr_cible].rank(method="min", ascending=False, na_option='bottom')

#Je suis pas mal : mon ranking est à moins de 1 du vrai ranking. C parce que la DGCL semble
# avoir carotte la commune d'Ablon-sur-Seine (code commune 94001)

elig_cible = "Eligible DSR part cible"

lastdata[elig_cible] = (lastdata[rang_indice_synthetique_cible]<=10000)
#Montant garantie : on n'a pas trop moyen de connaître 2017 par contre 2018 on l'a

dsr_cible_garantie_vieux = "Dotation de solidarité rurale - Cible#Montant de la garantie de sortie cible rétroactive"
dsr_cible_garantie_lastyear = "Dotation de solidarité rurale - Cible#Montant de la garantie de sortie cible des communes devenues inéligibles en 2019"
total_budget_dsr_fraction_cible = 323_780_451
total_to_attribute_cible = total_budget_dsr_fraction_cible - lastdata[dsr_cible_garantie_vieux].sum() - lastdata[dsr_cible_garantie_lastyear].sum() - 4_549_441 #unexplained dons


montant_cible = "DSR fraction cible Montant attribué éligible"
montant_cible_1 = montant_cible + " - part potentiel financier par habitant"
montant_cible_2 = montant_cible + " - part voirie"
montant_cible_3 = montant_cible + " - part enfants"
montant_cible_4 = montant_cible + " - part potentiel financier par hectare"


dsr_cible_montant_total = "DSR cible Montant attribué total"


# part 3.1 : potentiel financier
weight_dsr_cible_1 = 0.3
score_dsr_cible_1 = montant_cible_1.replace("Montant attribué", "score")

lastdata[score_dsr_cible_1] = (lastdata[elig_cible]
                          * lastdata[pop_dgf]
                          * np.minimum(lastdata[effort_fiscal], cap_effort_fiscal)
                          * lastdata[facteur_pot_fin_strate])
print("to attr", weight_dsr_cible_1*total_to_attribute_cible, "tot score", lastdata[score_dsr_cible_1].sum(), "VP :", weight_dsr_cible_1*total_to_attribute_cible / lastdata[score_dsr_cible_1].sum())
valeur_point_cible_1 = weight_dsr_cible_1*total_to_attribute_cible / lastdata[score_dsr_cible_1].sum()
lastdata[montant_cible_1] = lastdata[score_dsr_cible_1] * valeur_point_cible_1

# part 3.2 : voirie
weight_dsr_cible_2 = 0.3
score_dsr_cible_2 = montant_cible_2.replace("Montant attribué", "score")
lastdata[score_dsr_cible_2] = lastdata[elig_cible] * lastdata[longueur_voirie] * (1 + ((lastdata[is_montagne]=="OUI") | (lastdata[is_insulaire]=="OUI")))
print("to attr", weight_dsr_cible_2*total_to_attribute_cible, "tot score", lastdata[score_dsr_cible_2].sum(), "VP :", weight_dsr_cible_2*total_to_attribute_cible / lastdata[score_dsr_cible_2].sum())
valeur_point_cible_2 = weight_dsr_cible_2*total_to_attribute_cible / lastdata[score_dsr_cible_2].sum()
lastdata[montant_cible_2] = lastdata[score_dsr_cible_2] * valeur_point_cible_2

# part 3.3 : enfants
weight_dsr_cible_3 = 0.3
score_dsr_cible_3 = montant_cible_3.replace("Montant attribué", "score")
lastdata[score_dsr_cible_3] = lastdata[elig_cible] * lastdata[nb_hab_3_16]
print("to attr", weight_dsr_cible_3*total_to_attribute_cible, "tot score", lastdata[score_dsr_cible_3].sum(), "VP :", weight_dsr_cible_3*total_to_attribute_cible / lastdata[score_dsr_cible_3].sum())
valeur_point_cible_3 = weight_dsr_cible_3*total_to_attribute_cible / lastdata[score_dsr_cible_3].sum()
lastdata[montant_cible_3] = lastdata[score_dsr_cible_3] * valeur_point_cible_3

# part 3.4 : potentiel financier par hectare
weight_dsr_cible_4 = 0.1
score_dsr_cible_4 = montant_cible_4.replace("Montant attribué", "score")
lastdata[score_dsr_cible_4] = lastdata[elig_cible] * lastdata[pop_dgf] * np.maximum(0, 2 - lastdata[pot_fin_par_hectare] / lastdata[pot_fin_10000_hectare])
print("to attr", weight_dsr_cible_4*total_to_attribute_cible, "tot score", lastdata[score_dsr_cible_4].sum(), "VP :", weight_dsr_cible_4*total_to_attribute_cible / lastdata[score_dsr_cible_4].sum())
valeur_point_cible_4 = weight_dsr_cible_4*total_to_attribute_cible / lastdata[score_dsr_cible_4].sum()
lastdata[montant_cible_4] = lastdata[score_dsr_cible_4] * valeur_point_cible_4

lastdata[montant_cible] = lastdata[montant_cible_1] + lastdata[montant_cible_2] +lastdata[montant_cible_3] + lastdata[montant_cible_4]
print(lastdata[montant_cible].sum())
lastdata.loc[(lastdata[montant_previous_year_cible] > 0) & (lastdata[elig_cible]), montant_cible] = np.maximum(
    0.9 * lastdata[montant_previous_year_cible], lastdata[montant_cible])
print(lastdata[montant_cible].sum())
lastdata.loc[(lastdata[montant_previous_year_cible] > 0) & (lastdata[elig_cible]), montant_cible] = np.minimum(
    1.2 * lastdata[montant_previous_year_cible], lastdata[montant_cible])



lastdata[dsr_cible_montant_total] = lastdata[montant_cible] + lastdata[dsr_cible_garantie_vieux] + lastdata[dsr_cible_garantie_lastyear]


dsr_montant_total = "DSR - montant total attribué"

lastdata[dsr_montant_total] = lastdata[dsr_bc_montant_total] + lastdata[dsr_pq_montant_total] + lastdata[dsr_cible_montant_total]

actual_dsr_montant_total = "DSR - Vrai montant total attribué"

lastdata[actual_dsr_montant_total] = lastdata[actual_montant_dsr_bc] + lastdata[actual_montant_dsr_pq] + lastdata[actual_montant_dsr_cible]


#1/0
#cols_interessantes = [code_comme,pop_plaf, part_canton, bur_centr, chef_lieu_canton, is_chef_lieu_canton, elig_bc]

#Comparison report : giving information about the precision of the computation compared to some actual
#data we got from DGCL...

def comparison_report(actual_col,computed_col,comp_style="number",bool_names=["OUI",False]):
    if comp_style=="number":
        diff=lastdata[actual_col]-lastdata[computed_col]
        exsts=lastdata[actual_col] | lastdata[computed_col]
        totabsdiff=sum([abs(k) for k in diff])
        nbexists=len([k for k in exsts if k])
        print("Entre",actual_col,"et ",computed_col)
        print("nb different", len([k for k in diff if k]), "/ ", nbexists)
        print("avg diff", totabsdiff/nbexists, "avg diff relative", totabsdiff/lastdata[computed_col].sum())
    if comp_style=="bool":
        print(lastdata.groupby([actual_col,computed_col])[[code_comm]].count())
    print("*****Report donE*****")

actual_rang_indice_synthetique_cible = "Dotation de solidarité rurale - Cible#Rang DSR Cible"
actual_elig_cible = "Est éligible cible selon la DGCL"
lastdata[actual_elig_cible] = (lastdata[actual_rang_indice_synthetique_cible] >0) & (lastdata[actual_rang_indice_synthetique_cible] < nb_communes_cible)
actual_montant_elig_cible = "Vrai montant d eligibilité estimé cible"
lastdata[actual_montant_elig_cible] = lastdata[actual_montant_dsr_cible] * lastdata[actual_elig_cible]
actual_montant_elig_pq = "vrai montant d éligibilité estimé péréquation"
lastdata[actual_montant_elig_pq] = lastdata[actual_montant_dsr_pq] * lastdata[actual_elig_pq]

for comp in (
    [elig_bc,actual_elig_bc,"bool"],
    [elig_pq,actual_elig_pq,"bool"],
    [elig_cible,actual_elig_cible,"bool"],
    [dsr_bc_montant_eligible,actual_montant_elig_bc,"number"],
    [dsr_bc_montant_total,actual_montant_dsr_bc,"number"],
    [dsr_pq_montant_total,actual_montant_dsr_pq,"number"],
    [dsr_pq_montant_total,actual_montant_elig_pq,"number"],
    [dsr_cible_montant_total,actual_montant_dsr_cible,"number"],
    [montant_cible, actual_montant_elig_cible, "number"],
    [dsr_montant_total, actual_dsr_montant_total, "number"],
):
    comparison_report(comp[1],comp[0],comp[2])

"""
dsr_p="Dotation de solidarité rurale - Péréquation#Montant global réparti (après garantie CN)"
dsr_c="Dotation de solidarité rurale - Cible#Montant global réparti"
totalsum=0
for k in (dsr_bc,dsr_p,dsr_c):
    print(lastdata[k].sum())
    totalsum+=lastdata[k].sum()
print("=",totalsum)
synth=pd.read_pickle("synth{}.pkl".format(annee_calcul))
print(synth.sum())"""