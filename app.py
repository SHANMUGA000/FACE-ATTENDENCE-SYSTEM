from flask import Flask, render_template, request, Response, send_file
import sqlite3
from datetime import datetime
import threading
import subprocess
from io import BytesIO
from reportlab.pdfgen import canvas

app = Flask(__name__)

def start_attendance():
    # Add your code to start camera and take attendance here
    subprocess.run(['python', 'attendance_taker.py'])
    print("Attendance started!")

@app.route('/')
def index():
    current_date = datetime.now().strftime('%Y-%m-%d')
    return render_template('index.html', selected_date=current_date, no_data=False)

@app.route('/attendance', methods=['POST'])
def attendance():
    selected_date = request.form.get('selected_date')
    if not selected_date:
        selected_date = datetime.now().strftime('%Y-%m-%d')

    selected_date_obj = datetime.strptime(selected_date, '%Y-%m-%d')
    formatted_date = selected_date_obj.strftime('%Y-%m-%d')

    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()

    cursor.execute("SELECT name, student_reg_no, student_dept_code, student_degree_code, "
                               "student_current_sem_no, student_batch_no, in_time, out_time, date,sub_name, faculty_name, no_of_hours, timing, sub_code, Session FROM attendance WHERE date = ?", (formatted_date,))
    attendance_data = cursor.fetchall()
    print(attendance_data)
    conn.close()

    if not attendance_data:
        return render_template('index.html', selected_date=selected_date, no_data=True)

    return render_template('index.html', selected_date=selected_date, attendance_data=attendance_data)

@app.route('/start_attendance')
def start_attendance_route():
    threading.Thread(target=start_attendance).start()
    return "Attendance started!"

@app.route('/export_csv/<selected_date>')
def export_csv_route(selected_date):
    selected_date_obj = datetime.strptime(selected_date, '%Y-%m-%d')
    formatted_date = selected_date_obj.strftime('%Y-%m-%d')

    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()

    cursor.execute("SELECT name, student_reg_no, Session, student_dept_code, student_degree_code, student_current_sem_no, student_batch_no, in_time, out_time FROM attendance WHERE date = ?", (formatted_date,))
    attendance_data = cursor.fetchall()

    conn.close()

    if not attendance_data:
        return "No data to export"

    # Generate CSV content
    csv_content = "name,student_reg_no,Session,student_dept_code,student_degree_code,student_current_sem_no,student_batch_no,in_time,out_time\n"
    for entry in attendance_data:
        csv_content += ",".join(map(str, entry)) + "\n"

    # Set appropriate headers for CSV file
    headers = {
        'Content-Type': 'text/csv',
        'Content-Disposition': f'attachment; filename=attendance_{formatted_date}.csv'
    }

    return Response(csv_content, headers=headers)

@app.route('/export_pdf/<selected_date>')
def export_pdf_route(selected_date):
    selected_date_obj = datetime.strptime(selected_date, '%Y-%m-%d')
    formatted_date = selected_date_obj.strftime('%Y-%m-%d')

    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()

    cursor.execute("SELECT name, student_reg_no, Session, student_dept_code, student_degree_code, student_current_sem_no, student_batch_no, in_time, out_time FROM attendance WHERE date = ?", (formatted_date,))
    attendance_data = cursor.fetchall()
    conn.close()

    if not attendance_data:
        return "No data to export"

    # Generate PDF content
    pdf_filename = f'attendance_{formatted_date}.pdf'

    # Create PDF using Flask-Canvas
    pdf = canvas.Canvas(pdf_filename)

    pdf.drawString(100, 800, "Attendance Report - {}".format(formatted_date))
    y_position = 780
    for entry in attendance_data:
        pdf.drawString(100, y_position, ",".join(map(str, entry)))
        y_position -= 20

    # Save the PDF to BytesIO
    pdf_stream = BytesIO()
    pdf.save()

    # Set appropriate headers for PDF file
    headers = {
        'Content-Type': 'application/pdf',
        'Content-Disposition': f'attachment; filename={pdf_filename}'
    }

    # Write the PDF content to BytesIO
    pdf_stream.seek(0)
    pdf_stream.write(pdf.getpdfdata())
    pdf_stream.seek(0)

    return send_file(pdf_stream, as_attachment=True, download_name=pdf_filename)

if __name__ == '__main__':
    app.run(debug=True)
