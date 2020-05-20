from reportlab.pdfgen import canvas

class GeraPDF:
    def __init__(self):
        pass
    def tabela(self, nome_pdf):
        try:
            pdf = canvas.Canvas('{}.pdf'.format(nome_pdf))
            x = 720
            lista = {'Rafaela': '19', 'Jose': '15', 'Maria': '22','Eduardo':'24'}
            for nome,idade in lista.items():
                x -= 20
                pdf.drawString(247,x, '{} : {}'.format(nome,idade))
            pdf.setTitle(nome_pdf)
            pdf.setFont("Helvetica-Oblique", 14)
            pdf.drawString(245,750, 'Lista de Convidados')
            pdf.setFont("Helvetica-Bold", 12)
            pdf.drawString(245,724, 'Nome e idade')
            pdf.save()
            print('{}.pdf criado com sucesso!'.format(nome_pdf))
        except:
            print("ERRO")