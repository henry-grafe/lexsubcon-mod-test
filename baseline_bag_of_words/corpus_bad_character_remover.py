import regex as re

escape_illegal_xml_characters = lambda x: re.sub(u'[\x00-\x08\x0b\x0c\x0e-\x1F\uD800-\uDFFF\uFFFE\uFFFF]', '', x)

f_current = open("C:\\Users\\NICOLAS\\Documents\\KULeuven\\master_thesis\\datasets\\UKWAC\\UKWAC-1.xml\\UKWAC-1_processing_ready.xml",'r',encoding="latin5")
f_new = open("C:\\Users\\NICOLAS\\Documents\\KULeuven\\master_thesis\\datasets\\UKWAC\\UKWAC-1.xml\\UKWAC-1_parsing_ready.xml",'w',encoding="latin5")
print('starting rewriting...')
buffer = f_current.readline()
while buffer!="":
    buffer = escape_illegal_xml_characters(buffer)
    f_new.write(buffer)
    buffer = f_current.readline()
    for i in range(100):
        buffer += f_current.readline()
print('done !')
f_new.close()
f_current.close()