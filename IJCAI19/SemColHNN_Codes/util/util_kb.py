"""
This file contains functions of DBPedia lookup and SPARQL query
"""

import requests
import re
import time
import sparql
import xml.etree.ElementTree as ET

dbp_prefix = 'http://dbpedia.org/resource/'
dbo_prefix = 'http://dbpedia.org/ontology/'


# lookup top-k entities from DBPedia of a cell
# needed by cache_entities.py
def lookup_entities(cell_text, top_k):
    entities = list()
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
            lookup_url = 'http://lookup.dbpedia.org/api/search/KeywordSearch?MaxHits=%d&QueryString=%s' \
                         % (top_k, cell_item)
            lookup_res = requests.get(lookup_url)
            if '400 Bad Request' not in lookup_res.content:
                root = ET.fromstring(lookup_res.content)
                for child in root:
                    ent = child[1].text
                    entities.append(ent)
        except UnicodeDecodeError:
            pass
    return entities


# lookup top-k entities from DBPedia of a cell
def lookup_entities_with_repeat(cell_text, top_k):
    entities = list()
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
            lookup_url = 'http://lookup.dbpedia.org/api/search/KeywordSearch?MaxHits=%d&QueryString=%s' \
                         % (top_k, cell_item)
            lookup_res = requests.get(lookup_url)
            if '400 Bad Request' in lookup_res.content:
                time.sleep(60*5)
                lookup_res = requests.get(lookup_url)

            if '400 Bad Request' not in lookup_res.content:
                root = ET.fromstring(lookup_res.content)
                for child in root:
                    ent = child[1].text
                    entities.append(ent)
        except UnicodeDecodeError:
            pass
    return entities


# lookup top-k entities from DBPedia of a cell
# partial information (, \/, ()) are retrieved
def lookup_entities_with_partial(cell_text, top_k):
    cell_origin = cell_text
    entities = list()

    cell_brackets = re.findall('\((.*?)\)', cell_text)
    for cell_bracket in cell_brackets:
        cell_text = cell_text.replace('(%s)' % cell_bracket, '')
    if ',' in cell_text:
        cell_text = cell_text.replace(',', ' ')
    cell_text = cell_text.strip()

    try:
        lookup_url = 'http://lookup.dbpedia.org/api/search/KeywordSearch?MaxHits=%d&QueryString=%s' % (top_k, cell_text)
        lookup_res = requests.get(lookup_url)
        if '400 Bad Request' not in lookup_res.content:
            root = ET.fromstring(lookup_res.content)
            for child in root:
                ent = child[1].text
                entities.append(ent)
    except UnicodeDecodeError:
        pass

    if len(entities) < top_k and ',' in cell_origin:
        for s in cell_origin.split(','):
            s = s.strip()
            try:
                lookup_url = 'http://lookup.dbpedia.org/api/search/KeywordSearch?MaxHits=%d&QueryString=%s' % (2, s)
                lookup_res = requests.get(lookup_url)
                if '400 Bad Request' not in lookup_res.content:
                    root = ET.fromstring(lookup_res.content)
                    for child in root:
                        ent = child[1].text
                        if len(entities) < top_k:
                            entities.append(ent)
            except UnicodeDecodeError:
                pass

    if len(entities) < top_k and len(cell_brackets) > 0:
        for s in cell_brackets:
            if len(s) > 2:
                lookup_url = 'http://lookup.dbpedia.org/api/search/KeywordSearch?MaxHits=%d&QueryString=%s' % (2, s)
                lookup_res = requests.get(lookup_url)
                if '400 Bad Request' not in lookup_res.content:
                    root = ET.fromstring(lookup_res.content)
                    for child in root:
                        ent = child[1].text
                        if len(entities) < top_k:
                            entities.append(ent)

    return entities


# Query DBPedia for an entity's complete classes
# needed by cache_classes.py
def query_complete_classes_of_entity(ent):
    if ent.startswith(dbp_prefix):
        ent = ent.split(dbp_prefix)[1]
    classes = set()
    try:
        s = sparql.Service('http://dbpedia.org/sparql', "utf-8", "GET")

        statement = 'select distinct ?superclass where { <%s%s> rdf:type ?e. ' \
                    '?e rdfs:subClassOf* ?superclass. FILTER(strstarts(str(?superclass), "%s"))}' \
                    % (dbp_prefix, ent, dbo_prefix)
        result = s.query(statement)
        for row in result.fetchone():
            cls_uri = str(row[0])
            cls = cls_uri.split(dbo_prefix)[1]
            classes.add(cls)

        statement = 'select distinct ?ss where {<%s%s> dbo:wikiPageRedirects ?e. ?e rdf:type ?s. ' \
                    '?s rdfs:subClassOf* ?ss. FILTER(strstarts(str(?ss), "%s"))}' \
                    % (dbp_prefix, ent, dbo_prefix)
        result = s.query(statement)
        for row in result.fetchone():
            cls_uri = str(row[0])
            cls = cls_uri.split(dbo_prefix)[1]
            classes.add(cls)
    except UnicodeDecodeError:
        print('     %s: UnicodeDecodeError' % ent)
        pass
    return classes


# Query DBPedia for an entity's complete classes
def query_complete_classes_of_entity_with_repeat(ent):
    if ent.startswith(dbp_prefix):
        ent = ent.split(dbp_prefix)[1]
    classes = set()
    try:
        s = sparql.Service('http://dbpedia.org/sparql', "utf-8", "GET")

        statement = 'select distinct ?superclass where { <%s%s> rdf:type ?e. ' \
                    '?e rdfs:subClassOf* ?superclass. FILTER(strstarts(str(?superclass), "%s"))}' \
                    % (dbp_prefix, ent, dbo_prefix)
        try:
            result = s.query(statement)
        except sparql.SparqlException:
            time.sleep(60 * 5)
            result = s.query(statement)

        for row in result.fetchone():
            cls_uri = str(row[0])
            cls = cls_uri.split(dbo_prefix)[1]
            classes.add(cls)

        statement = 'select distinct ?ss where {<%s%s> dbo:wikiPageRedirects ?e. ?e rdf:type ?s. ' \
                    '?s rdfs:subClassOf* ?ss. FILTER(strstarts(str(?ss), "%s"))}' \
                    % (dbp_prefix, ent, dbo_prefix)
        try:
            result = s.query(statement)
        except sparql.SparqlException:
            time.sleep(60 * 5)
            result = s.query(statement)

        for row in result.fetchone():
            cls_uri = str(row[0])
            cls = cls_uri.split(dbo_prefix)[1]
            classes.add(cls)
    except UnicodeDecodeError:
        print('     %s: UnicodeDecodeError' % ent)
        pass
    return classes


# given a class, find out all the properties of the entities of the class
# return pairs of <property, the number of entities>
# needed by properties.py
def query_property_ent_num(cls):
    prop_ent_num = dict()
    s = sparql.Service('http://dbpedia.org/sparql', "utf-8", "GET")
    statement = 'select ?pro (count(distinct ?ent) as ?ent_num) where {?ent rdf:type ?t. ?t rdfs:subClassOf* <%s%s>. ' \
                '?ent ?pro ?obj. FILTER(strstarts(str(?pro), "%s"))} group by ?pro' \
                % (dbo_prefix, cls, dbo_prefix)
    result = s.query(statement)
    for row in result.fetchone():
        p = str(row[0])
        n = int(str(row[1]))
        prop_ent_num[p] = n
    return prop_ent_num


# query the number of entities of a class
# needed by properties.py
def query_ent_num(cls):
    s = sparql.Service('http://dbpedia.org/sparql', "utf-8", "GET")
    statement = 'select count(?ent) where {?ent rdf:type ?t. ?t rdfs:subClassOf* <%s%s>}' % (dbo_prefix, cls)
    result = s.query(statement)
    for row in result.fetchone():
        return int(str(row[0]))


# query the property and object according to a subject entity
# a candidate set of properties are given
# needed by prop2vec.py
def query_property_object(ent, candidate_prop):
    pro_obj_s = []
    s = sparql.Service('http://dbpedia.org/sparql', "utf-8", "GET")
    statement = 'select ?p,?o where {<%s> ?p ?o. FILTER(strstarts(str(?p), "%s"))}' % (ent, dbo_prefix)
    try:
        result = s.query(statement)
    except sparql.SparqlException:
        time.sleep(60*5)
        result = s.query(statement)

    for row in result.fetchone():
        try:
            p = str(row[0])
            if p in candidate_prop:
                pro_obj_s.append([p, row[1]])
        except UnicodeEncodeError:
            pass
    return pro_obj_s
