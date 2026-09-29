from zipfile import ZipFile, ZIP_DEFLATED
from pathlib import Path, PurePosixPath
from lxml import etree as E
from copy import deepcopy
import posixpath,json,hashlib
B=Path('artifacts/presentation/.build')
SOURCE=Path('docs/materials/ЛЦТ2026 Шаблон презентации.pptx')
P='http://schemas.openxmlformats.org/presentationml/2006/main'; A='http://schemas.openxmlformats.org/drawingml/2006/main';R='http://schemas.openxmlformats.org/officeDocument/2006/relationships';PR='http://schemas.openxmlformats.org/package/2006/relationships'; CT='http://schemas.openxmlformats.org/package/2006/content-types'
NS={'p':P,'a':A,'r':R,'pr':PR,'ct':CT}
def xml(b):return E.fromstring(b)
def dump(x):return E.tostring(x,xml_declaration=True,encoding='UTF-8',standalone=True)
def relpath(part):p=PurePosixPath(part);return str(p.parent/'_rels'/(p.name+'.rels'))
def resolve(part,target):return target.lstrip('/') if target.startswith('/') else posixpath.normpath(posixpath.join(posixpath.dirname(part),target))
with ZipFile(SOURCE) as z: data={n:z.read(n) for n in z.namelist()}
with ZipFile(B/'authored.pptx') as z: authored={n:z.read(n) for n in z.namelist()}
# Targeted filling. No shape, geometry, theme, image, layout, or run style is recreated.
filled={7:{'740199916':'[НАЗВАНИЕ\nКОМАНДЫ]','757351073':'Кейс 08. Сервис прогнозирования инцидентов\nи управления ремонтными работами инженерных коллекторов Москвы'},
8:{'328740113':'[НАЗВАНИЕ\nКОМАНДЫ]','1075761414':'Риск тревожной записи по объекту за 24–48 часов. Приоритет проверки для диспетчера.','441518184':'Причинные признаки, сравнение с простыми правилами и воспроизводимые эксперименты.'},
9:{'415042907':'КОМАНДА'},
10:{'865297653':'О КОМАНДЕ','1588890124':'[Добавить историю команды и опыт совместной работы]','1574440249':'[Добавить причину выбора задачи командой]','243118211':'Проверили горизонт и отделили тревогу от аварии.\nИсключили будущие признаки. Сравнили правила,\nбустинг и GRU на одинаковых временных периодах.'},
11:{'1782181813':'Дневные признаки только из прошлого журнала.\nLightGBM оценивает риск тревожной записи на объекте за 24–48 часов.\nAPI и локальный интерфейс показывают объектный риск.\nПроверки причинности и точного воспроизведения прогнозов.',
'321655550':'Первый сценарий: приоритет проверок для диспетчера.\nПилот свяжет тревоги с итогами осмотров и стоимостью решений.\nИнтеграция с рабочими системами заказчика потребует отдельного этапа.\nЭкономический эффект пока не измерен.'}}
def set_shape_text(sp,text):
 tx=sp.find('p:txBody',NS); ps=tx.findall('a:p',NS)
 first=ps[0] if ps else E.Element('{'+A+'}p')
 rpr=first.find('a:r/a:rPr',NS)
 if rpr is None:rpr=first.find('a:endParaRPr',NS)
 ppr=first.find('a:pPr',NS)
 for p in ps:tx.remove(p)
 for line in text.split('\n'):
  p=E.SubElement(tx,'{'+A+'}p')
  if ppr is not None:p.append(deepcopy(ppr))
  r=E.SubElement(p,'{'+A+'}r')
  if rpr is not None:
   pr=deepcopy(rpr);pr.tag='{'+A+'}rPr';r.append(pr)
  E.SubElement(r,'{'+A+'}t').text=line
  if rpr is not None:
   pr=deepcopy(rpr);pr.tag='{'+A+'}endParaRPr';p.append(pr)
for n,changes in filled.items():
 part=f'ppt/slides/slide{n}.xml';root=xml(data[part]);seen=set()
 for sp in root.findall('.//p:sp',NS):
  ident=sp.find('p:nvSpPr/p:cNvPr',NS).get('id')
  if ident in changes:set_shape_text(sp,changes[ident]);seen.add(ident)
  if n==8 and ident=='661119785':
   for t in sp.findall('.//a:t',NS):
    swaps={'ФИО, специальность':'[ФИО, специальность]','__ человек':'[число] человек','как образовалась команда? ':'[История команды]','место работы/учебы участников?':'[Место работы/учёбы]','Город и регион:':'Город и регион: [заполнить]'}
    t.text=swaps.get(t.text,t.text)
  if n==9:
   for t in sp.findall('.//a:t',NS):
    if t.text=='Имя Фамилия':t.text='[Имя Фамилия]'
 assert seen==set(changes)
 data[part]=dump(root)
 # Notes remain native and carry source/provenance without invented team facts.
 nr=relpath(part)
 if nr in data:
  for rel in xml(data[nr]):
   if rel.get('Type','').endswith('/notesSlide'):
    note=resolve(part,rel.get('Target'));nrroot=xml(data[note]);body=nrroot.find('.//p:sp[p:nvSpPr/p:nvPr/p:ph[@type="body"]]',NS) if False else None
    for sp in nrroot.findall('.//p:sp',NS):
     ph=sp.find('p:nvSpPr/p:nvPr/p:ph',NS)
     if ph is not None and ph.get('type')=='body':set_shape_text(sp,'Источники: reports/object_forward_check/report.md, reports/object_sequence_candidate/report.md, docs/materials/8. ДЖКХ.pdf. Сведения о составе и истории команды не предоставлены. Геометрия обязательного слайда сохранена из оригинала.')
    data[note]=dump(nrroot)
# New slides were authored through @oai/artifact-tool. Isolate their package parts.
def authored_part(name):
 if name.startswith('ppt/slides/slide') and name.endswith('.xml'):
  index=int(name.split('/slide')[-1][:-4])
  return f'ppt/slides/slide{index if index<=6 else index+5}.xml'
 if '/_rels/' in name and name.endswith('.rels'):
  owner=name.replace('/_rels/','/')[:-5]
  return relpath(authored_part(owner))
 return 'ppt/authored/'+name
for name,blob in authored.items():
 if name=='[Content_Types].xml' or name=='_rels/.rels':continue
 if name.endswith('.rels'):
  owner=name.replace('/_rels/','/')[:-5]
  rr=xml(blob)
  for r in rr:
   if r.get('TargetMode')!='External':
    target=authored_part(resolve(owner,r.get('Target')))
    r.set('Target',posixpath.relpath(target,posixpath.dirname(authored_part(owner))))
  blob=dump(rr)
 data[authored_part(name)]=blob
pres=xml(data['ppt/presentation.xml']);rels=xml(data['ppt/_rels/presentation.xml.rels'])
for r in list(rels):
 if r.get('Type','').endswith('/slide'):rels.remove(r)
sl=pres.find('p:sldIdLst',NS)
for e in list(sl):sl.remove(e)
order=[f'ppt/slides/slide{i}.xml' for i in range(1,16)]
for n,part in enumerate(order,1):
 rid=f'rDeckSlide{n}';E.SubElement(rels,'{'+PR+'}Relationship',Id=rid,Type=R+'/slide',Target=posixpath.relpath(part,'ppt'))
 E.SubElement(sl,'{'+P+'}sldId',id=str(1000+n),attrib={'{'+R+'}id':rid})
# Register authored masters so both PowerPoint and Artifact Tool retain inheritance.
ar=xml(authored['ppt/_rels/presentation.xml.rels'])
for r in ar:
 suffix=r.get('Type','').split('/')[-1]
 if suffix not in ('slideMaster','notesMaster'):continue
 target=authored_part(resolve('ppt/presentation.xml',r.get('Target')))
 rid='rAuthored'+suffix+str(len(rels))
 E.SubElement(rels,'{'+PR+'}Relationship',Id=rid,Type=r.get('Type'),Target=posixpath.relpath(target,'ppt'))
 listtag='sldMasterIdLst' if suffix=='slideMaster' else 'notesMasterIdLst'
 itemtag='sldMasterId' if suffix=='slideMaster' else 'notesMasterId'
 dest=pres.find('p:'+listtag,NS)
 if dest is None:dest=E.SubElement(pres,'{'+P+'}'+listtag)
 at={'{'+R+'}id':rid}
 if suffix=='slideMaster':at['id']=str(2147483649+len(dest))
 E.SubElement(dest,'{'+P+'}'+itemtag,attrib=at)
# Clear source section/custom-show bookkeeping, if present.
for e in list(pres):
 if E.QName(e).localname in ('custShowLst','extLst'):pres.remove(e)
data['ppt/presentation.xml']=dump(pres);data['ppt/_rels/presentation.xml.rels']=dump(rels)
ct=xml(data['[Content_Types].xml']);act=xml(authored['[Content_Types].xml'])
new_overrides={authored_part(e.get('PartName').lstrip('/')) for e in act if e.get('PartName')}
for e in list(ct):
 if e.get('PartName','').lstrip('/') in new_overrides:ct.remove(e)
known={e.get('Extension') for e in ct if E.QName(e).localname=='Default'}
for e in act:
 e=deepcopy(e)
 if E.QName(e).localname=='Override':e.set('PartName','/'+authored_part(e.get('PartName').lstrip('/')));ct.append(e)
 elif e.get('Extension') not in known:ct.append(e);known.add(e.get('Extension'))
# Prune unreachable old instructional slides and resources by package relationships.
reachable=set()
def visit(part):
 if part in reachable:return
 if part not in data:raise ValueError('Missing part '+part)
 reachable.add(part)
 rp=relpath(part) if part else '_rels/.rels'
 if rp in data:
  reachable.add(rp)
  for r in xml(data[rp]):
   if r.get('TargetMode')!='External':visit(resolve(part,r.get('Target')))
reachable.add('_rels/.rels')
for r in xml(data['_rels/.rels']):
 if r.get('TargetMode')!='External':visit(resolve('',r.get('Target')))
for e in list(ct):
 if e.get('PartName') and e.get('PartName').lstrip('/') not in reachable:ct.remove(e)
data['[Content_Types].xml']=dump(ct);reachable.add('[Content_Types].xml')
with ZipFile(B/'candidate.pptx','w',ZIP_DEFLATED) as z:
 for part in sorted(reachable):z.writestr(part,data[part])
# Content-independent structural proof for all five compulsory slides.
checks=[]
with ZipFile(SOURCE) as src:
 for n in range(7,12):
  before=xml(src.read(f'ppt/slides/slide{n}.xml'));after=xml(data[f'ppt/slides/slide{n}.xml'])
  for root in (before,after):
   for tx in root.findall('.//p:txBody',NS):
    for para in list(tx):
     if E.QName(para).localname=='p':tx.remove(para)
  equal=E.tostring(before,method='c14n')==E.tostring(after,method='c14n')
  assert equal
  checks.append({'slide':n,'all_non_text_structure_exact':equal,'position_retained':n})
(B/'mandatory_fidelity.json').write_text(json.dumps({'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'slides':checks,'final_slide_order':order},ensure_ascii=False,indent=2))
print('Merged 15 slides with original compulsory slide structure')
