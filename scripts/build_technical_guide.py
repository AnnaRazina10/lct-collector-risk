#!/usr/bin/env python3
"""Render the seven-page Markdown guide with embedded Cyrillic fonts.

Requires reportlab. Fonts: --font-dir, CODEX runtime, or system DejaVu.
The source supports headings, paragraphs, links, tables, lists and code blocks.
Explicit page separators make the reviewed pagination reproducible.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import re

from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, KeepTogether, PageBreak, PageTemplate, Paragraph,
    Preformatted, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
NAVY = colors.HexColor('#163349')
TEAL = colors.HexColor('#087f83')
GRAY = colors.HexColor('#506674')
PALE = colors.HexColor('#eef5f7')
rl_config.invariant = 1


def font_directory(explicit):
    candidates = [Path(explicit)] if explicit else []
    candidates += [Path('/usr/share/fonts/truetype/dejavu')]
    runtime = Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies'
    candidates += [runtime/'native/libreoffice-headless/libreoffice/LibreOfficeDev.app/Contents/Resources/fonts/truetype']
    for path in candidates:
        if all((path/f).is_file() for f in ['DejaVuSans.ttf','DejaVuSans-Bold.ttf','DejaVuSansMono.ttf']):
            return path
    raise FileNotFoundError('Cyrillic DejaVu fonts missing; pass --font-dir')


def register_fonts(path):
    for name, filename in [('Guide','DejaVuSans.ttf'),('GuideBold','DejaVuSans-Bold.ttf'),('GuideMono','DejaVuSansMono.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(path/filename)))
    pdfmetrics.registerFontFamily('Guide',normal='Guide',bold='GuideBold',italic='Guide',boldItalic='GuideBold')


def inline(text):
    """Escape text and translate only documented inline markup, never raw HTML."""
    tokens=[]
    pattern=r'\[([^\]]+)\]\(((?:https?://|\.\./)[^)]+)\)|`([^`]+)`|\*\*([^*]+)\*\*'
    end=0
    for match in re.finditer(pattern,text):
        tokens.append(html.escape(text[end:match.start()]))
        if match.group(1):
            tokens.append(f'<link href="{html.escape(match.group(2),quote=True)}" color="#087f83"><u>{html.escape(match.group(1))}</u></link>')
        elif match.group(3):
            tokens.append(f'<font name="GuideMono">{html.escape(match.group(3))}</font>')
        else:tokens.append('<b>'+html.escape(match.group(4))+'</b>')
        end=match.end()
    tokens.append(html.escape(text[end:]))
    return ''.join(tokens)


def styles():
    body=ParagraphStyle('Body',fontName='Guide',fontSize=9.6,leading=13.6,textColor=NAVY,spaceAfter=7,alignment=TA_LEFT,allowWidows=0,allowOrphans=0)
    return {
        'body':body,
        'h1':ParagraphStyle('H1',parent=body,fontName='GuideBold',fontSize=20,leading=24,spaceBefore=0,spaceAfter=13,keepWithNext=True),
        'h2':ParagraphStyle('H2',parent=body,fontName='GuideBold',fontSize=13,leading=17,spaceAfter=11,keepWithNext=True,textColor=TEAL),
        'h3':ParagraphStyle('H3',parent=body,fontName='GuideBold',fontSize=11,leading=15,spaceBefore=9,spaceAfter=7,keepWithNext=True),
        'table':ParagraphStyle('Table',parent=body,fontSize=8.6,leading=11.8,spaceAfter=0),
        'tablehead':ParagraphStyle('TableHead',parent=body,fontName='GuideBold',fontSize=8.7,leading=11.8,spaceAfter=0,textColor=colors.white),
        'code':ParagraphStyle('Code',parent=body,fontName='GuideMono',fontSize=7.8,leading=11,spaceAfter=0),
        'quote':ParagraphStyle('Quote',parent=body,fontSize=9.5,leading=13.6,spaceAfter=0),
    }


def table_widths(header,width):
    count=len(header)
    if count==4:
        proportions=[.15,.35,.15,.35] if header[0]=='Год' else [.39,.19,.21,.21]
    elif count==3:
        proportions=[.50,.25,.25] if header[0].startswith('Показатель') else [.24,.38,.38]
    else:
        proportions=[.40,.60] if header[0]=='API' else [.24,.76]
    return [width*v for v in proportions]


def convert_page(source,style,width):
    result=[];lines=source.strip().splitlines();index=0
    while index<len(lines):
        line=lines[index].strip()
        if not line:index+=1;continue
        if line.startswith('```'):
            index+=1;code=[]
            while index<len(lines) and not lines[index].startswith('```'):
                code.append(lines[index]);index+=1
            for item in code:
                if pdfmetrics.stringWidth(item,'GuideMono',7.8)>width-20:
                    raise ValueError('Code line exceeds page width: '+item)
            block=Preformatted('\n'.join(code),style['code'])
            t=Table([[block]],colWidths=[width]);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),PALE),('LEFTPADDING',(0,0),(-1,-1),10),('RIGHTPADDING',(0,0),(-1,-1),10),('TOPPADDING',(0,0),(-1,-1),9),('BOTTOMPADDING',(0,0),(-1,-1),9)]))
            result.extend([t,Spacer(1,9)]);index+=1;continue
        if line.startswith('|'):
            rows=[]
            while index<len(lines) and lines[index].strip().startswith('|'):
                cells=[c.strip() for c in lines[index].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?',c.replace(' ','')) for c in cells):rows.append(cells)
                index+=1
            widths=table_widths(rows[0],width)
            cells=[[Paragraph(inline(c),style['tablehead' if r==0 else 'table']) for c in row] for r,row in enumerate(rows)]
            t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
            commands=[('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),5.5),('BOTTOMPADDING',(0,0),(-1,-1),5.5),('LINEBELOW',(0,0),(-1,0),.5,NAVY)]
            for row in range(1,len(cells)):
                if row%2:commands.append(('BACKGROUND',(0,row),(-1,row),PALE))
                commands.append(('LINEBELOW',(0,row),(-1,row),.25,colors.HexColor('#d7e3e9')))
            t.setStyle(TableStyle(commands));result.extend([t,Spacer(1,9)]);continue
        if line.startswith('#'):
            level=len(line)-len(line.lstrip('#'))
            result.append(Paragraph(inline(line[level:].strip()),style['h'+str(min(level,3))]));index+=1;continue
        if line.startswith('> '):
            t=Table([[Paragraph(inline(line[2:]),style['quote'])]],colWidths=[width])
            t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),PALE),('LINEBEFORE',(0,0),(-1,-1),3,TEAL),('LEFTPADDING',(0,0),(-1,-1),12),('TOPPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),10)]))
            result.extend([t,Spacer(1,9)]);index+=1;continue
        paragraph=[line];index+=1
        is_list=bool(re.match(r'^\d+\.\s',line))
        while index<len(lines) and lines[index].strip() and not re.match(r'^(#|\||```|> |\d+\.\s)',lines[index].strip()):
            paragraph.append(lines[index].strip());index+=1
        result.append(Paragraph(inline(' '.join(paragraph)),style['body']))
    return result


def build(source,output,font_dir):
    register_fonts(font_dir);style=styles()
    text=source.read_text(encoding='utf-8')
    if any(c in text for c in '\u2010\u2011\u2012\u2013\u2014'):
        raise ValueError('Use ASCII hyphens in the source')
    sections=text.split('<!-- page -->')
    if len(sections)!=7:raise ValueError('Expected seven explicitly paginated sections')
    width,height=A4;margin=17*mm;usable=width-2*margin
    class GuideDoc(BaseDocTemplate):
        def afterFlowable(self,flowable):
            if isinstance(flowable,Paragraph) and flowable.style.name=='H1':
                key=f'section-{self.page}'
                self.canv.bookmarkPage(key)
                self.canv.addOutlineEntry(flowable.getPlainText(),key,0,False)
    def chrome(canvas,doc):
        canvas.saveState();canvas.setStrokeColor(TEAL);canvas.setLineWidth(1)
        canvas.line(margin,height-14*mm,width-margin,height-14*mm)
        canvas.setFont('Guide',7.5);canvas.setFillColor(GRAY)
        canvas.drawString(margin,height-11*mm,'ЛЦТ 2026  /  КЕЙС 08  /  СОПРОВОДИТЕЛЬНОЕ РУКОВОДСТВО')
        canvas.line(margin,15*mm,width-margin,15*mm)
        canvas.drawString(margin,10.7*mm,'Коллектор: очередь риска  |  29.09.2026  |  bd0fa01')
        canvas.drawRightString(width-margin,10.7*mm,f'{doc.page} / 7')
        canvas.restoreState()
    output.parent.mkdir(parents=True,exist_ok=True)
    doc=GuideDoc(str(output),pagesize=A4,leftMargin=margin,rightMargin=margin,topMargin=20*mm,bottomMargin=20*mm,title='Коллектор: очередь риска. Техническое руководство',author='lct-collector-risk',pageCompression=1)
    frame=Frame(margin,20*mm,usable,height-40*mm,id='main',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
    doc.addPageTemplates(PageTemplate(id='guide',frames=[frame],onPage=chrome))
    story=[]
    for i,section in enumerate(sections):
        if i:story.append(PageBreak())
        story.extend(convert_page(section,style,usable))
    doc.build(story)
    if doc.page!=7:raise ValueError(f'Layout overflow: expected 7 pages, got {doc.page}')
    print(json.dumps({'pdf':str(output),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'pdf_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'bytes':output.stat().st_size,'font_dir':str(font_dir)},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'docs/technical_guide.md')
    parser.add_argument('--output',type=Path,default=ROOT/'output/pdf/collector-risk-technical-guide.pdf')
    parser.add_argument('--font-dir',default=os.environ.get('COLLECTOR_GUIDE_FONT_DIR'))
    args=parser.parse_args();build(args.source,args.output,font_directory(args.font_dir))
