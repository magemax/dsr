from typing import Dict, List, Optional
import os

import pandas  # type: ignore
import numpy as np

from openfisca_core.simulation_builder import SimulationBuilder  # type: ignore
from openfisca_france_dotations_locales import CountryTaxBenefitSystem  # type: ignore


TBS = CountryTaxBenefitSystem()


elig_bc_dgcl = "Eligible fraction bourg-centre selon DGCL"
elig_pq_dgcl = "Eligible fraction péréquation selon DGCL"
elig_cible_dgcl = "Eligible fraction cible selon DGCL"
code_comm = "Informations générales - Code INSEE de la commune"
nom_comm = "Informations générales - Nom de la commune"




### Start load_data_dgcl

variables_openfisca_presentes_fichier ={
    'bureau_centralisateur': 'Dotation de solidarité rurale Bourg-centre - Bureaux centralisateurs',
    'chef_lieu_arrondissement': "Dotation de solidarité rurale Bourg-centre - Chef-lieu d'arrondissement au 31 décembre 2014",
    'chef_lieu_de_canton': 'Dotation de solidarité rurale Bourg-centre - Code commune chef-lieu de canton au 1er janvier 2014',
    'chef_lieu_departement_dans_agglomeration': 'Dotation de solidarité rurale Bourg-centre - Chef-lieu de département agglo',
    'part_population_canton': 'Dotation de solidarité rurale Bourg-centre - Pourcentage de la population communale dans le canton',
    'population_dgf': "Informations générales - Population DGF Année N'",
    'population_dgf_agglomeration': "Dotation de solidarité rurale Bourg-centre - Population DGF des communes de l'agglomération",
    'population_dgf_departement_agglomeration': "Dotation de solidarité rurale Bourg-centre - Population départementale de référence de l'agglomération",
    'population_dgf_plafonnee': 'Dotation de solidarité rurale Bourg-centre - Population DGF plafonnée',
    'population_insee': 'Informations générales - Population INSEE Année N ',
    'potentiel_financier': 'Potentiel fiscal et financier des communes - Potentiel financier',
    'potentiel_financier_par_habitant': 'Potentiel fiscal et financier des communes - Potentiel financier par habitant',
    'revenu_total': 'Dotation de solidarité urbaine - Revenu imposable des habitants de la commune',
    'strate_demographique': 'Informations générales - Strate démographique Année N'
}

# A partir de l'adresse du tableau publié par la DGCL, produit un tableau contenant toutes les colonnes nécessaires
# au calcul des dotations.
# il a deux sources de données : 
# - des colonnes déjà présentes dans le fichier (qu'il suffit de renommer), leur nom
# est présent dans le dictionnaire variables_openfisca_presentes_fichier. On 
# met aussi tout ce qui apparait dans autres_cols_interessantes (pour qu'un export)
# de data soit sympa;
# - des colonnes qu'il faut construire, et qui sont ajoutées petit à petit
def load_dgcl_file( path="2019-communes-criteres-repartition.csv"):
    data = pandas.read_csv(path, decimal= ",")
    extracolumns={}
    #
    # add_plus_grande_commune_agglo_column
    #
    # Niveau ceinture blanche : on groupe by la taille totale de l'agglo (c'est ce qu'on fait icis)
    # Niveau ceinture orange : en plus de ce critère, utiliser des critères géographiques pour localiser les agglos.
    # Ceinture rouge : on trouve des données sur les agglos de référence de chaque commune
    pop_agglo=variables_openfisca_presentes_fichier["population_dgf_agglomeration"]
    pop_dgf = variables_openfisca_presentes_fichier["population_dgf"]
    max_pop_plus_grosse_commune = data.groupby(pop_agglo)[[pop_dgf]].max()
    # Ca marche car le plus haut nombre d'habitants à être partagé par 2 agglo est 23222
    # print(somme_pop_plus_grosse_commune[somme_pop_plus_grosse_commune.index!=somme_pop_plus_grosse_commune[somme_pop_plus_grosse_commune.columns[0]]])
    plus_grosse_commune = "Population plus grande commune de l'agglomération"
    max_pop_plus_grosse_commune.columns = [plus_grosse_commune]
    max_pop_plus_grosse_commune.loc[max_pop_plus_grosse_commune.index == 0, plus_grosse_commune] = 0
    data=data.merge(max_pop_plus_grosse_commune,left_on=pop_agglo,right_index=True).sort_index()
    extracolumns["population_dgf_maximum_commune_agglomeration"] = plus_grosse_commune
    
    #
    # Corrige la valeur du département : 
    #
    #pop_departement = variables_openfisca_presentes_fichier["population_dgf_departement_agglomeration"]
    #data[pop_departement] = np.maximum(data[pop_departement], 1)

    #
    # introduit la valeur d'outre mer
    #
    departement = "Informations générales - Code département de la commune"
    data[departement]=data[departement].astype(str)
    outre_mer_dgcl= "commune d'outre mer"
    data[outre_mer_dgcl] = (data[departement].str.len()>2)  & (~data[departement].str.contains("A")) & (~data[departement].str.contains("B"))
    extracolumns["outre_mer"]=outre_mer_dgcl

    #
    # Chope les infos du chef-lieu de canton
    #
    is_chef_lieu_canton = "Chef-lieu de canton"
    chef_lieu_de_canton_dgcl="Dotation de solidarité rurale Bourg-centre - Code commune chef-lieu de canton au 1er janvier 2014"
    #data[chef_lieu_de_canton_dgcl]=data[chef_lieu_de_canton_dgcl].apply(lambda x: str(x).zfill(5))
    data[is_chef_lieu_canton]= (data[chef_lieu_de_canton_dgcl]==data[code_comm])
    extracolumns["chef_lieu_de_canton"] = is_chef_lieu_canton

    table_chef_lieu_canton=data[[code_comm,pop_dgf]]
    pop_dgf_chef_lieu_canton= pop_dgf+" du chef-lieu de canton"
    table_chef_lieu_canton.columns=[code_comm, pop_dgf_chef_lieu_canton]
    data=data.merge(table_chef_lieu_canton,left_on=chef_lieu_de_canton_dgcl,right_on=code_comm,how="left", suffixes=("","_Chef_lieu"))
    data[pop_dgf_chef_lieu_canton] = data[pop_dgf_chef_lieu_canton].fillna(0).astype(int)
    extracolumns["population_dgf_chef_lieu_de_canton"]=pop_dgf_chef_lieu_canton

    # Corrige les infos sur le revenu total de la commune quand il est à 0 et que l'indice synthétique est renseigné
    # Certains revenus moyens sont missing...
    # On essaye de les remplir grâce à notre super equation:
    # RT = pop_insee * (0.3*RMStrate)/(IS-0.7 * PF(strate)/PF)
    #pour ceci, calculons le revenu moyen de chaque strate
    strate= variables_openfisca_presentes_fichier["strate_demographique"]
    revenu_total_dgcl = variables_openfisca_presentes_fichier["revenu_total"]
    pop_insee = variables_openfisca_presentes_fichier["population_insee"]
    tableau_donnees_par_strate= data[(~data[outre_mer_dgcl])].groupby(strate)[[pop_insee,revenu_total_dgcl]].sum()
    revenu_moyen_strate= " Revenu imposable moyen par habitant de la strate"
    tableau_donnees_par_strate[revenu_moyen_strate] = tableau_donnees_par_strate[revenu_total_dgcl]/tableau_donnees_par_strate[pop_insee]
    # print(tableau_donnees_par_strate)
    data = data.merge(tableau_donnees_par_strate[[revenu_moyen_strate]],left_on=strate,right_index=True)
    
    # Avant de corriger les revenus, il nous faut calculer les revenus moyens par strate
    # Certains revenus sont ignorés dans le calcul du revenu moyen de la strate, on sait pas pourquoi
    # (ptet la DGCL préserve le secret statistique dans ses metrics agrégées?)
    # La conséquence de la ligne de code qui suit est qu'on utilise la même méthodo que la DGCL
    # donc on a un classement cible plus proche de la vérité.
    # L'inconvénient est que les revenus moyens par strate ne sont pas reproductibles en l'état
    # On pourrait rajouter une variable dans Openfisca qui dirait "est ce que cette commune est 
    # prise en compte dans la moyenne de revenu moyen par strate?" , mais le calcul de cette variable 
    # n'est pas dans la loi.  
    extracolumns["revenu_par_habitant_moyen"]=revenu_moyen_strate
    

    actual_indice_synthetique = "Dotation de solidarité rurale - Cible - Indice synthétique"
    revenu_moyen_strate = " Revenu imposable moyen par habitant de la strate"
    pot_fin_strate = "Potentiel fiscal et financier des communes - Potentiel financier moyen de la strate"
    pot_fin_par_hab = "Potentiel fiscal et financier des communes - Potentiel financier par habitant"
    # print("communes sans revenu avec habitants before correction")
    # print(len(data[(data[revenu_total_dgcl]==0) & (data[pop_insee]>0)]))
    data.loc[(data[revenu_total_dgcl]==0) & (data[pop_insee]>0) & (data[actual_indice_synthetique]>0), revenu_total_dgcl]= (
        0.3 * data[revenu_moyen_strate] / (data[actual_indice_synthetique] - 0.7 * data[pot_fin_strate]/data[pot_fin_par_hab])
    ) * data[pop_insee]
    # print("communes sans revenu avec habitants after correction")
    # print(len(data[(data[revenu_total_dgcl]==0) & (data[pop_insee]>0)]))
    
    #
    # Génère le dataframe au format final :
    # Il contient toutes les variables qu'on rentrera dans openfisca au format openfisca 
    # + Des variables utiles qu'on ne rentre pas dans openfisca
    #
    # Restriction aux colonnes intéressantes : 
    rang_indice_synthetique = "Dotation de solidarité rurale - Cible - Rang DSR Cible"
    translation_cols={**variables_openfisca_presentes_fichier,**extracolumns}
    data[elig_bc_dgcl] = (data["Dotation de solidarité rurale Bourg-centre - Montant de la commune éligible"]>0)
    data[elig_pq_dgcl] = (data["Dotation de solidarité rurale - Péréquation - Part Pfi (avant garantie CN)"]>0)
    data[elig_cible_dgcl] = (data[rang_indice_synthetique] >0) & (data[rang_indice_synthetique] <=10000)
    autres_cols_interessantes = [code_comm, nom_comm,rang_indice_synthetique, actual_indice_synthetique, elig_bc_dgcl, elig_pq_dgcl, elig_cible_dgcl ]
    data = data[autres_cols_interessantes + list(translation_cols.values())]

    # Renomme colonnes
    invert_dict={name_dgcl:name_ofdl for name_ofdl,name_dgcl in translation_cols.items()}
    data.columns= [column if column not in invert_dict else invert_dict[column] for column in data.columns]
    # Passe les "booléens dgf" (oui/non) en booléens normaux
    liste_columns_to_real_bool= ["bureau_centralisateur", "chef_lieu_arrondissement", "chef_lieu_departement_dans_agglomeration"]
    for col in liste_columns_to_real_bool:
        data[col] = (data[col].str.contains(pat="oui", case=False))
    return data


### End load_data_dgcl

### Start Generate simulation

def simulation_from_dgcl_csv(period, data, tbs):
    sb=SimulationBuilder()
    sb.create_entities(tbs)
    sb.declare_person_entity("commune", data.index)
    simulation = sb.build(tbs)
    for champ_openfisca in data.columns:
        if " " not in champ_openfisca: # oui c'est comme ça que je checke
        # qu'une variable es openfisca ne me jugez pas
            simulation.set_input(
                champ_openfisca,
                period,
                data[champ_openfisca],
            )
    return simulation
### load data
### fait tourner 1 sim normale
### fait tourner 1 sim reformée
### outputte un dataframe

#prend un dictionnaire de reformes (apres, eventuellement PLF)

def resultfromreforms(dict_ref = {}, to_compute_res= ["dsr_eligible_fraction_bourg_centre", "dsr_eligible_fraction_perequation", "dsr_eligible_fraction_cible"]):
    PERIOD="2020"
    #some of these can be preloaded in memory to improve performance.
    DATA=load_dgcl_file()
    dict_sims = {"avant":simulation_from_dgcl_csv(PERIOD, DATA, TBS)}

    for nom_scenario, reforme_scenario in dict_ref.items():
        TBS_Modified = DotationReform(TBS,reforme_scenario)
        dict_sims[nom_scenario]=simulation_from_dgcl_csv(PERIOD, DATA, TBS_Modified)

    d2=DATA
    for nom_scenario,sim in dict_sims.items():
        for tc in to_compute_res:
            d2[tc+"_"+nom_scenario]=sim.calculate(tc,PERIOD)
    d2.to_csv("resultsenfin.csv")
    return d2

### End simulation



# La réforme : prendre un dictionnaire décrivant la réforme et la changer en TBS.

### Reforme

#Retourne un dictionnaire flattened à partir d'un nested dictionnaire
def flattened_dict( dict_to_traverse,prechain = "", delimiter="."):
    res= {}
    pc = [prechain] if len(prechain) else []
    for k,v in dict_to_traverse.items():
        if isinstance(v,dict):
            res={**res,**flattened_dict(v, prechain=delimiter.join(pc+[k]))}
        else:
            res[delimiter.join(pc+[k])]=v
    return res

def test_flattened_dict():
    a={"a":{"b":{"c":3, "d":{"e":4,"f":"chaton"}}}}
    b={"a.b.c":3,"a.b.d.e":4,"a.b.c.d.f":"chaton"}
    assert(flattened_dict(a)==b)



from openfisca_core import periods
from openfisca_core.reforms import Reform  # type: ignore
from openfisca_core.parameters import ParameterNode  # type: ignore
from functools import reduce


class DotationReform(Reform):

    def __init__(self, tbs, payload: dict) -> None:
        self.payload = payload.get("dgf", {})
        self.period = periods.period("year:1900:200")
        super().__init__(tbs)

    def modifier(self, parameters: ParameterNode) -> ParameterNode:
        #passe en revue tous les champs du dictionnaire de réforme de payload.
        #Ceux qui ne sont pas des dictionnaires seront mis à ce niveau dans ofdl
        flatpayload=flattened_dict(self.payload)
        print(flatpayload)
        for field_to_update,value in flatpayload.items():
            print("updating", field_to_update)
            reduce(getattr,field_to_update.split("."),parameters).update(period=self.period, value= value)
        
        return parameters

    def apply(self) -> None:
        self.modify_parameters(modifier_function=self.modifier)

payload_example ={
    "dgf":{
        "dotation_solidarite_rurale":{
            "cible":{
                "eligibilite":{
                    "seuil_classement":23
                }
            }
        }
    }
}


### End Reforme

### compute_impact
    
def impacts_reforme_dotation(reforme= payload_example):
    variables_nombre_communes= ["dsr_eligible_fraction_bourg_centre", "dsr_eligible_fraction_perequation", "dsr_eligible_fraction_cible"]
    to_compute = variables_nombre_communes
    df_results=resultfromreforms({"apres" : reforme}, to_compute)
    res={}
    for scenario in ["avant", "apres"]:
        res[scenario]={}
        for col in variables_nombre_communes:
            res[scenario][col] = df_results[col+"_"+scenario].sum()
    return res

### end_compute_impact

### TODO : Dataframe to res : transforme le resultat d'impacts_reforme_dotation en resultat renvoyable.

### TODO : methode pour traduire le payload reçu en payload format openfisca
###

### TODO : somehow wraph this shite into an API.

#aux compare Shit
tocompare={"dsr_eligible_fraction_bourg_centre":elig_bc_dgcl, 
        "dsr_eligible_fraction_perequation":elig_pq_dgcl, 
        "dsr_eligible_fraction_cible":elig_cible_dgcl}

def check_variables_bool(data):
    res={}
    for k,v in tocompare.items():
        res[k]=data.pivot_table(code_comm, index=k,columns=v, aggfunc="count", fill_value=0)
    return res


def print_eligilble_comparison():
    PERIOD="2020"
    DATA=load_dgcl_file()
    sim=simulation_from_dgcl_csv(PERIOD, DATA, TBS)
    colonnes_to_compare=["dsr_eligible_fraction_bourg_centre", 
        "dsr_eligible_fraction_perequation", 
        "dsr_eligible_fraction_cible"]
    D2=DATA

    for variable_to_compute in colonnes_to_compare+["indice_synthetique_dsr_cible","rang_indice_synthetique_dsr_cible"]:
        DATA[variable_to_compute]=sim.calculate(variable_to_compute, PERIOD)

    DATA.to_csv("data_compare.csv")
    nbt=0
    nbT=0

    for nom_ofdl, resultats_comparaison in check_variables_bool(DATA).items():
        print("Comparaison DGCL vs nous pour le calcul de",nom_ofdl)
        print(resultats_comparaison)

### end aux compare shit

if __name__=="__main__":
    from time import time
    st=time()
    print_eligilble_comparison()
    print("Elapsed : {:.2f}".format(time()-st))
    st=time()
    print(impacts_reforme_dotation())
    print("Elapsed : {:.2f}".format(time()-st))





