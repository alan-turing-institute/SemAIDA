import requests
import re
import sparql
import xml.etree.ElementTree as ET

dbp_prefix = 'http://dbpedia.org/resource/'
dbo_prefix = 'http://dbpedia.org/ontology/'


# lookup entities and classes from DBPedia
def lookup_resources(cell_text):
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
            lookup_url = 'http://lookup.dbpedia.org/api/search/KeywordSearch?MaxHits=2&QueryString=%s' % cell_item
            lookup_res = requests.get(lookup_url)
            root = ET.fromstring(lookup_res.content)
            for child in root:
                ent = child[1].text.split(dbp_prefix)[1]
                entities.append(ent)
        except UnicodeDecodeError:
            pass
    return entities


# lookup dbo_classes of matched entity by cell text
def lookup_dbo_classes(cell_text):
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
def query_general_entities(cls_entities, seeds_num):
    cls_gen_entities = dict()
    s = sparql.Service('http://dbpedia.org/sparql', "utf-8", "GET")
    for cls in cls_entities.keys():
        par_entities = cls_entities[cls]
        entities = list()
        statement = 'select distinct ?e where {?e a <%s%s>} ORDER BY RAND() limit %d' % (dbo_prefix, cls, seeds_num)
        result = s.query(statement)
        for row in result.fetchone():
            ent_uri = str(row[0])
            ent = ent_uri.split(dbp_prefix)[1]
            if ent not in par_entities:
                entities.append(ent)
        cls_gen_entities[cls] = entities
        print('%s done, %d entities' % (cls, len(entities)))
    return cls_gen_entities


# Query DBPedia for a class' super classes
def query_super_classes(cls):
    s = sparql.Service('http://dbpedia.org/sparql', "utf-8", "GET")
    supers = set()
    statement = 'SELECT distinct ?superclass WHERE { <%s%s> rdfs:subClassOf* ?superclass. ' \
                'FILTER ( strstarts(str(?superclass), "%s"))}' % (dbo_prefix, cls, dbo_prefix)
    result = s.query(statement)
    for row in result.fetchone():
        super_str = str(row[0])
        super_name = super_str.split(dbo_prefix)[1]
        supers.add(super_name)

    return supers


# Query DBPedia for a class' children classes
def query_sub_classes(cls):
    s = sparql.Service('http://dbpedia.org/sparql', "utf-8", "GET")
    subs = set()
    statement = 'SELECT distinct ?s WHERE { ?s rdfs:subClassOf* <%s%s>. ' \
                'FILTER ( strstarts(str(?s), "%s"))}' % (dbo_prefix, cls, dbo_prefix)
    result = s.query(statement)
    for row in result.fetchone():
        sub_str = str(row[0])
        sub_name = sub_str.split(dbo_prefix)[1]
        subs.add(sub_name)

    return subs


# Query DBPedia for an entity's complete classes
def query_complete_classes_of_entity(ent):
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
    except UnicodeDecodeError:
        print('     %s: UnicodeDecodeError' % ent)
        pass
    return classes


# Query DBPedia for a class' equivalent classes
def equivalent_classes(cls):
    classes = set()
    s = sparql.Service('http://dbpedia.org/sparql', "utf-8", "GET")
    statement = 'SELECT distinct ?eqclass WHERE { <%s%s> owl:equivalentClass ' \
                '?eqclass. FILTER ( strstarts(str(?eqclass), "%s"))}' % (dbo_prefix, cls, dbo_prefix)
    result = s.query(statement)
    for row in result.fetchone():
        cls_uri = str(row[0])
        cls = cls_uri.split(dbo_prefix)[1]
        classes.add(cls)
    return classes


# return fine-grained classes among a set of classes
def fine_grained_class(cs):
    fg_c = set()
    for c in cs:
        sub_classes = query_sub_classes(c)
        if len(cs.intersection(sub_classes)) <= 1:
            fg_c.add(c)
    return fg_c
