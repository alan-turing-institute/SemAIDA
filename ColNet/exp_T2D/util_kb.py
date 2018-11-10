"""
This file includes functions related to Dbpedia lookup and SPARQL endpoint queries
"""
import requests
import re
import sparql
import xml.etree.ElementTree as ET


# lookup entities and classes from DBPedia
def lookup_resources(cell_text):
    dbo_prefix = 'http://dbpedia.org/ontology/'
    dbp_prefix = 'http://dbpedia.org/resource/'
    entity_classes = dict()
    cell_items = list()
    cell_brackets = re.findall('\((.*?)\)', cell_text)
    for cell_bracket in cell_brackets:
        cell_text = cell_text.replace('(%s)' % cell_bracket, '')
    cell_text = cell_text.strip()
    if len(cell_text) > 2:
        cell_items.append(cell_text)
    for cell_bracket in cell_brackets:
        if len(cell_bracket) > 2:
            cell_items.append(cell_bracket.strip())
    for cell_item in cell_items:
        try:
            lookup_url = 'http://lookup.dbpedia.org/api/search/KeywordSearch?MaxHits=2&QueryString=%s' % cell_item
            lookup_res = requests.get(lookup_url)
            root = ET.fromstring(lookup_res.content)
            for child in root:
                entity = child[1].text.split(dbp_prefix)[1]
                classes = list()
                for cc in child[3]:
                    cls_URI = cc[1].text
                    if dbo_prefix in cls_URI:
                        classes.append(cls_URI.split(dbo_prefix)[1])
                entity_classes[entity] = classes
        except UnicodeDecodeError:
            pass
    return entity_classes


# lookup dbo_classes of matched entity by cell text
def lookup_dbo_classes(cell_text):
    dbo_prefix = 'http://dbpedia.org/ontology/'
    cell_items = list()
    cell_brackets = re.findall('\((.*?)\)', cell_text)
    for cell_bracket in cell_brackets:
        cell_text = cell_text.replace('(%s)' % cell_bracket, '')
    cell_text = cell_text.strip()
    if len(cell_text) > 2:
        cell_items.append(cell_text)
    for cell_bracket in cell_brackets:
        if len(cell_bracket) > 2:
            cell_items.append(cell_bracket.strip())
    for cell_item in cell_items:
        try:
            lookup_url = 'http://lookup.dbpedia.org/api/search/KeywordSearch?MaxHits=1&QueryString=%s' % cell_item
            lookup_res = requests.get(lookup_url)
            root = ET.fromstring(lookup_res.content)
            for child in root:
                classes = set()
                for c in child[3]:
                    cls_uri = c[1].text
                    if dbo_prefix in cls_uri:
                        classes.add(cls_uri.split(dbo_prefix)[1])
                return classes
        except UnicodeDecodeError:
            pass
    return set()


# Query DBPedia
def query_general_entities(cls_entities):
    dbp_prefix = 'http://dbpedia.org/resource/'
    cls_gen_entities = dict()
    s = sparql.Service('http://dbpedia.org/sparql', "utf-8", "GET")
    for cls in cls_entities.keys():
        par_entities = cls_entities[cls]
        entities = list()
        statement = 'select distinct ?e where {?e a dbo:%s} ORDER BY RAND() limit 1000' % cls
        result = s.query(statement)
        for row in result.fetchone():
            ent_uri = str(row[0])
            ent = ent_uri.split(dbp_prefix)[1]
            if ent not in par_entities:
                entities.append(ent)
        cls_gen_entities[cls] = entities
        print('%s done, %d entities' % (cls, len(entities)))
    return cls_gen_entities


# extend table's classes with super classes
def super_classes(col_classes):
    dbo_prefix = 'http://dbpedia.org/ontology/'
    s = sparql.Service('http://dbpedia.org/sparql', "utf-8", "GET")
    for i, col in enumerate(col_classes.keys()):
        ori_cls = col_classes[col][0]
        statement = 'SELECT distinct ?superclass WHERE { dbo:%s rdfs:subClassOf* ?superclass. ' \
                    'FILTER ( strstarts(str(?superclass), "%s"))}' % (ori_cls, dbo_prefix)
        result = s.query(statement)
        for row in result.fetchone():
            super_cls = str(row[0])
            super_cls_name = super_cls.split(dbo_prefix)[1]
            if super_cls_name not in col_classes[col]:
                col_classes[col].append(super_cls_name)
        if i % 10 == 0:
            print('%d columns done' % (i + 1))
    return col_classes


# extend classes with their super classes
def super_classes_of_classes(classes):
    super_clses = dict()
    dbo_prefix = 'http://dbpedia.org/ontology/'
    s = sparql.Service('http://dbpedia.org/sparql', "utf-8", "GET")
    for cls in classes:
        supers = set()
        statement = 'SELECT distinct ?superclass WHERE { dbo:%s rdfs:subClassOf* ?superclass. ' \
                    'FILTER ( strstarts(str(?superclass), "%s"))}' % (cls, dbo_prefix)
        result = s.query(statement)
        for row in result.fetchone():
            super_str = str(row[0])
            super_name = super_str.split(dbo_prefix)[1]
            if super_name not in supers:
                supers.add(super_name)

        super_clses[cls] = supers

    return super_clses
