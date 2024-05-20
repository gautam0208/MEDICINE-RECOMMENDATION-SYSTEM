from flask import Flask, request, render_template,redirect,url_for,session,send_from_directory, jsonify, send_file  # Import jsonify
import os
import numpy as np
import pandas as pd
import pickle
from flask_mysqldb import MySQL
import MySQLdb.cursors
import json

# flask app
app = Flask(__name__)



# load databasedataset===================================
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv('datasets/medications.csv')
diets = pd.read_csv("datasets/diets.csv")


# load model===========================================
svc = pickle.load(open('models/svc.pkl','rb'))


#============================================================
# custome and helping functions
#==========================helper funtions================
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == dis] ['workout']


    return desc,pre,med,die,wrkout

symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer disease',
                 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine',
                 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox',
                 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D',
                 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia',
                 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism',
                 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis',
                 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection',
                 35: 'Psoriasis', 27: 'Impetigo'}

# Model Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]


##database connection
app.secret_key = 'thisisthesecretkey'


app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'medicine_recommendation'
app.config['UPLOAD_FOLDER']='uploads'

mysql = MySQL(app)

# creating routes========================================

@app.route("/")
def homepage():
    return render_template("homepage.html")

@app.route('/pLogin')
def pLogin():
    return render_template("user_login.html",msg="Login Before Predicting your diseases!")

@app.route('/user_register')
def user_register():
    return render_template("user_register.html")

@app.route('/user_login')
def user_login():
    return render_template("user_login.html")

@app.route('/user_signup',methods=['POST','GET'])
def user_signup():
     msg=''
     if request.method=='POST' and 'name' in request.form and 'email_no' in request.form and 'password1' in request.form:
         name=request.form['name']
         email_no = request.form['email_no']
         psw = request.form['password1']
         cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
         cursor.execute('SELECT * FROM user_register WHERE Email_Phone_No = % s', (email_no,))
         account = cursor.fetchone()
         if account:
             msg = 'Account already exists !'
         else:
             cursor.execute('INSERT INTO user_register VALUES \
           		(NULL, % s, % s, % s)', (name, email_no, psw,))
             mysql.connection.commit()
             msg1 = 'You have successfully registered !'
             return render_template('user_login.html', register_msg=msg1)
     elif request.method == 'POST':
          msg = 'Please fill out the form !'
     return render_template('user_register.html', msg=msg)


@app.route('/user_signin',methods=['POST','GET'])
def user_signin():
    msg2=''
    if request.method=='POST' and 'email_no' in request.form and 'password' in request.form:
        email_no=request.form['email_no']
        password=request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM user_register WHERE Email_Phone_No = % s AND Password = % s', (email_no, password,))
        account = cursor.fetchone()
        if account:
            session['loggedin'] = True
            session['uid'] = account['ID']
            session['user_name'] = account['Name']

            msg2 = 'Logged in successfully !'

            return render_template('index.html',user_name=session['user_name'], login_msg=msg2)
        else:
            msg2 = 'Incorrect email/phone_no or password !'
    elif request.method == 'POST':
        msg2 = 'Please fill out the form !'
    return render_template('user_login.html', msg=msg2)

@app.route('/logout_user')
def logout_user():
 session.pop('loggedin', None)
 session.pop('uid', None)
 session.pop('name', None)
 return redirect(url_for('homepage'))


@app.route("/index")
def index():
    return render_template("index.html",user_name=session['user_name'])

# Define a route for the home page
@app.route('/predict', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        symptoms = request.form.get('selected_symptoms')
        if symptoms =="":
            msg = "No Symptoms to Predict!"
            return render_template('index.html', msg=msg)
        else:

            # Split the user's input into a list of symptoms (assuming they are comma-separated)

            symptoms=symptoms.strip();
            user_symptoms = [s.strip() for s in symptoms.split(',')]
            user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]

            predicted_disease = get_predicted_value(user_symptoms)
            dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)

            my_precautions = []
            for i in precautions[0]:
                my_precautions.append(i)

            my_med = ""
            for i in medications:
                my_med += i + ","
            m =[m.strip("[]' ") for m in my_med.split(",")]
            x=len(m)-1
            m=m[:x]

            my_diet = ""
            for i in rec_diet:
                my_diet += i + ","
            d= [d.strip("[]' ") for d in my_diet.split(",")]
            y=len(d)-1
            d=d[:y]

            low=['allergy','peptic ulcer disease','gastroenteritis','bronchial asthma','migraine','cervical spondylosis',
                 'common cold','dimorphic hemmorhoids(piles)','acne','chicken pox','urinary tract infection','psoriasis','impetigo']
            medium=['gerd','fungal infection','chronic cholestasis','hypertension','jaundice','hypothyroidism','hyperthyroidism','hypoglycemia',
                    'osteoarthritis','arthritis','(vertigo) paroysmal positional vertigo']
            high=['aids','diabetes','drug reaction','paralysis (brain hemorrhage)','malaria','dengue','typhoid','hepatitis a','hepatitis b',
                  'hepatitis c','hepatitis d','hepatitis e','alcoholic hepatitis','tuberculosis','pneumonia','heart attack','varicose veins']

            severity=''
            if predicted_disease.lower() in low:
                severity='low'
            elif predicted_disease.lower() in medium:
                severity='medium'
            elif predicted_disease.lower() in high:
                severity='high'

            return render_template('result.html', predicted_disease=predicted_disease, dis_des=dis_des,
                                   my_precautions=my_precautions, medications=m, my_diet=d,workout=workout,msg="",severity=severity)

    return render_template('index.html')



# about view funtion and path
@app.route('/about')
def about():
    return render_template("about.html")
# contact view funtion and path
@app.route('/contact')
def contact():
    return render_template("contact.html")

# developer view funtion and path
@app.route('/developer')
def developer():
    return render_template("developer.html")

# about view funtion and path
@app.route('/blog')
def blog():
    return render_template("blog.html")

@app.route('/services')
def services():
    return render_template("services.html")

@app.route('/result')
def result():
    return render_template("result.html")

#footer
@app.route('/about_footer')
def about_footer():
    return render_template("about_footer.html")

@app.route('/contact_footer')
def contact_footer():
    return render_template("contact_footer.html")


@app.route('/developer_footer')
def developer_footer():
    return render_template("developer_footer.html")


@app.route('/blog_footer')
def blog_footer():
    return render_template("blog_footer.html")

@app.route('/services_footer')
def services_footer():
    return render_template("services_footer.html")

@app.route('/appointment')
def appointment():
    return render_template("appointment.html")

@app.route('/func_patient_register',methods=['POST','GET'])
def patient_register():
    msg=''
    msg1=''
    if request.method=='POST' and 'fname' in request.form and 'lname' in request.form and 'email' in request.form and 'contact' in request.form and 'password' in request.form and 'gender' in request.form:
      fname=request.form['fname']
      lname = request.form['lname']
      email = request.form['email']
      contact = request.form['contact']
      password = request.form['password']
      gender = request.form['gender']
      cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
      cursor.execute('SELECT * FROM patient_register WHERE Email = % s || Phone_Number=% s', (email,contact,) )
      account = cursor.fetchone()
      if account:
          msg = 'Account already exists !'
      else:
          cursor.execute('INSERT INTO patient_register VALUES \
      		(NULL, % s, % s, % s, % s, % s,% s)',(fname, lname, email,contact, gender, password,))
          mysql.connection.commit()
          msg1 = 'You have successfully registered !'
          return render_template('patient_login.html',msg=msg1)
    elif request.method == 'POST':
        msg = 'Please fill out the form !'
    return render_template('appointment.html', msg=msg)



@app.route('/patient_login')
def patient_login():
    return render_template("patient_login.html")

specs = []
docDict = {}
dfees = {}
appHistory=[]
@app.route('/func_patient_login',methods=['POST','GET'])
def patientLogin():
    msg2=''
    if request.method=='POST' and 'email_phno' in request.form and 'password' in request.form:
        email_phno=request.form['email_phno']
        password=request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM patient_register WHERE (Email = % s OR Phone_Number=% s) AND Password = % s', (email_phno,email_phno, password,))
        account = cursor.fetchone()
        if account:
            session['loggedin'] = True
            session['pid'] = account['ID']
            session['fname'] = account['FirstName']
            session['lname'] = account['LastName']
            session['email'] = account['Email']
            session['contact'] = account['Phone_Number']
            session['gender'] = account['Gender']

            msg2 = 'Logged in successfully !'

           #fetching doctor details
            cursor.execute('SELECT  * FROM add_doctor')
            docdata = cursor.fetchall()

            for row in docdata:
             if specs.count(row['Specialization'])==0:
                specs.append(row['Specialization'])

            for i in specs:
                docDict[i]=[]

            for row in docdata:
                docDict[row['Specialization']].append(row['Doctor_Name']+" - "+row['Employee_ID'])
                dfees[row['Doctor_Name']+" - "+row['Employee_ID']+" "+row['Specialization']]=row['Fees']

           #fetching appointment details
            cursor.execute('SELECT  * FROM book_appointment')
            appointment=cursor.fetchall()

            cursor.execute('SELECT * from book_appointment WHERE Approved="true" ')
            approve = cursor.fetchall()
            cursor.execute('SELECT * from patient_description WHERE  Submitted="true"')
            submitted = cursor.fetchall()
            cursor.execute('SELECT * from prescription WHERE  Prescribed="true"')
            prescribe = cursor.fetchall()
            cursor.execute('SELECT * from patient_description')
            patient_desc = cursor.fetchall()
            cursor.execute('SELECT * from prescription')
            prescription = cursor.fetchall()

            cursor.execute('SELECT * from add_doctor where Available=true')
            available = cursor.fetchall()

            return render_template('patient_dashboard.html', msg=msg2,pid=session['pid'], name=session['fname'],appHistory=appointment,
                                   specialization=specs,doctor=docDict,fees=dfees,approve=approve,submitted=submitted,
                                   prescribe=prescribe,patient_desc=patient_desc,prescription=prescription,available=available)
        else:
            msg2 = 'Incorrect email/phone no OR Password !'
    return render_template('patient_login.html', msg=msg2)



@app.route('/patient_dashboard')
def patient_dashboard():
    return render_template('patient_dashboard.html')

@app.route('/book_appointment',methods=['POST','GET'])
def book_appointment():
    appMsg=''
    approve=''
    submitted=''
    if request.method=='POST' and 'spec' in request.form and 'doctor' in request.form and 'docFees' in request.form and 'appdate' in request.form and 'apptime' in request.form:
         special=request.form['spec']
         docNameID=request.form['doctor']
         fees=request.form['docFees']
         date=request.form['appdate']
         time=request.form['apptime']

         doc=docNameID.split(' - ')

         docName=doc[0]
         docEmpID=doc[1]


         cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
         cursor.execute('SELECT * FROM book_appointment WHERE Specialization = % s AND DOC_EMP_ID=% s AND Date = % s AND Time=% s',
                        (special, docEmpID, date,time,))
         account = cursor.fetchone()

         cursor.execute('SELECT  * FROM add_doctor where Specialization=% s and Employee_ID=% s',(special,docEmpID,))
         docId =cursor.fetchone();

         # fetching appointment details
         cursor.execute('SELECT  * FROM book_appointment')
         appointment = cursor.fetchall()

         # fetching doctor details
         cursor.execute('SELECT  * FROM add_doctor')
         docdata = cursor.fetchall()

         for row in docdata:
             if specs.count(row['Specialization']) == 0:
                 specs.append(row['Specialization'])

         for i in specs:
             docDict[i] = []

         for row in docdata:
             docDict[row['Specialization']].append(row['Doctor_Name'] + " - " + row['Employee_ID'])
             dfees[row['Doctor_Name'] + " - " + row['Employee_ID'] + " " + row['Specialization']] = row['Fees']

         cursor.execute('SELECT  * FROM book_appointment')
         appointment = cursor.fetchall()
         cursor.execute('SELECT * from book_appointment WHERE Approved="true" ')
         approve = cursor.fetchall()
         cursor.execute('SELECT * from patient_description WHERE  Submitted="true"')
         submitted = cursor.fetchall()
         cursor.execute('SELECT * from prescription WHERE  Prescribed="true"')
         prescribe = cursor.fetchall()
         cursor.execute('SELECT * from patient_description')
         patient_desc = cursor.fetchall()
         cursor.execute('SELECT * from prescription')
         prescription = cursor.fetchall()
         cursor.execute('SELECT * from add_doctor where Available=true')
         available = cursor.fetchall()
         if account:
             app_fail_Msg="Apointment Already Booked!"

             return render_template('patient_dashboard.html',pid=session['pid'],name=session['fname'],app_fail_Msg=app_fail_Msg
                                    ,appHistory=appointment,specialization=specs,doctor=docDict,fees=dfees,approve=approve,
                                    submitted=submitted,prescribe=prescribe,patient_desc=patient_desc,prescription=prescription,available=available)
         else:
             flag='false'
             cursor.execute('INSERT INTO book_appointment VALUES (NULL, % s,% s,% s,% s,% s,%s ,% s ,% s,% s,% s,% s,% s,% s,% s)',
                            (session['pid'],session['fname'],session['lname'],session['email'],session['contact'],session['gender'],docId['ID'], special,docName,docEmpID,fees,date,time,flag,))
             mysql.connection.commit()

             app_success_Msg="Appointment Booked!"
             # fetching appointment details


             return render_template('patient_dashboard.html',pid=session['pid'],name=session['fname'], app_success_Msg=app_success_Msg,
                                    appHistory=appointment,specialization=specs,doctor=docDict,fees=dfees,approve=approve,
                                    submitted=submitted,prescribe=prescribe,patient_desc=patient_desc,prescription=prescription,available=available)
    elif request.method=='POST':
        appMsg="fill out the form completely"
    return render_template('patient_dashboard.html',pid=session['pid'],name=session['fname'],appMsg=appMsg,specialization=specs,
                           doctor=docDict,fees=dfees,approve=approve,submitted=submitted)


@app.route('/patient_desc',methods=['POST','GET'])
def patient_desc():
    msg=''
    approve=''
    submitted=''
    prescribe=''
    patient_desc=''
    prescription=''
    appointment=''
    available=''
    if request.method=='POST' and  'symp' in request.form:
        symp=request.form['symp']
        complication=request.form['comp']
        dis=request.form['dis']
        file=request.files['reports']
        pid_appid=request.form['pat_app_id']
        pid_appid=pid_appid.split(" ")
        pId=pid_appid[0]
        appId=pid_appid[1]


       # file.save('uploads/'+file.filename)

        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute("Select * from patient_description where Patient_ID=% s and Appointment_ID=% s",(pId,appId,))
        account=cursor.fetchone()

        cursor.execute('SELECT  * FROM add_doctor')
        docdata = cursor.fetchall()

        for row in docdata:
            if specs.count(row['Specialization']) == 0:
                specs.append(row['Specialization'])

        for i in specs:
            docDict[i] = []

        for row in docdata:
            docDict[row['Specialization']].append(row['Doctor_Name'] + " - " + row['Employee_ID'])
            dfees[row['Doctor_Name'] + " - " + row['Employee_ID'] + " " + row['Specialization']] = row['Fees']

            # fetching appointment details
            cursor.execute('SELECT  * FROM book_appointment')
            appointment = cursor.fetchall()

            cursor.execute('SELECT * from book_appointment WHERE Approved="true" ')
            approve = cursor.fetchall()
            cursor.execute('SELECT * from patient_description WHERE  Submitted="true"')
            submitted = cursor.fetchall()
            cursor.execute('SELECT * from prescription WHERE  Prescribed="true"')
            prescribe = cursor.fetchall()
            cursor.execute('SELECT * from patient_description')
            patient_desc = cursor.fetchall()
            cursor.execute('SELECT * from prescription')
            prescription = cursor.fetchall()
            cursor.execute('SELECT * from add_doctor where Available=true')
            available = cursor.fetchall()

        if account:
            return render_template('patient_dashboard.html', desc_msg=msg, pid=session['pid'], name=session['fname'],
                                   appId=appId, appHistory=appointment, specialization=specs, doctor=docDict,
                                   fees=dfees, approve=approve,submitted=submitted, prescribe=prescribe, patient_desc=patient_desc,prescription=prescription, available=available)
        else:
         cursor.execute('INSERT INTO patient_description VALUES (NULL,% s,% s,% s,% s,% s,% s,% s)',
                             (pId,appId,symp,complication,dis,file.filename,"true",))
         file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

         mysql.connection.commit()

         msg="Details Submitted to Doctor!"
        return render_template('patient_dashboard.html', desc_msg=msg,pid=session['pid'],name=session['fname'],appId=appId,appHistory=appointment,specialization=specs,doctor=docDict,fees=dfees,approve=approve,
                               submitted=submitted,prescribe=prescribe,patient_desc=patient_desc,prescription=prescription,available=available)


@app.route('/download',methods=['POST','GET'])
def get_file():
    if request.method=='POST' and 'app_val' in request.form:
        app_val = request.form['app_val']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute("SELECT File_Name FROM patient_description WHERE Appointment_ID = %s ", (app_val,))
        file = cursor.fetchone()
        fileRoute = 'uploads/' + file['File_Name']
        return send_file(fileRoute, as_attachment=True)
    elif request.method == 'POST' and 'appoint_val' in request.form:
     appVal = request.form['appoint_val']
     cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
     cursor.execute("SELECT File_Name FROM patient_description WHERE Appointment_ID = %s ", (appVal,))
     file = cursor.fetchone()
     fileRoute = 'uploads/' + file['File_Name']
     return send_file(fileRoute, as_attachment=True)

@app.route('/doctor_login',methods=['POST','GET'])
def doctor_login():
    msg=''
    if request.method=='POST' and 'doc_empid_email' in request.form and 'doc_password' in request.form:
        doc_empid_email=request.form['doc_empid_email']
        password=request.form['doc_password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM add_doctor WHERE (Employee_ID = % s OR Email = % s) and password=% s', (doc_empid_email,doc_empid_email,password,))
        account = cursor.fetchone()
        if account:
            session['loggedin'] = True
            session['doc_id'] = account['ID']
            session['spec']=account['Specialization']
            session['doc_name'] = account['Doctor_Name']
            session['emp_id'] = account['Employee_ID']
            msg="Logged in successful"

            cursor.execute('Update add_doctor SET Available=true WHERE Employee_ID=% s',(session['emp_id'],))

            cursor.execute('SELECT  * FROM book_appointment')
            appointment = cursor.fetchall()

            pid=[]
            for row in appointment:
                pid.append(row['Patient_ID'])

            cursor.execute('SELECT * from book_appointment WHERE Approved="true" ')
            approve = cursor.fetchall()
            cursor.execute('SELECT * from patient_description WHERE  Submitted="true"')
            submitted = cursor.fetchall()
            cursor.execute('SELECT * from prescription WHERE  Prescribed="true"')
            prescribe = cursor.fetchall()

            cursor.execute('SELECT * from patient_description')
            patient_desc = cursor.fetchall()
            cursor.execute('SELECT * from prescription')
            prescription = cursor.fetchall()
            mysql.connection.commit()
            return render_template('doctor_dashboard.html',msg=msg,doc_name=session['doc_name'],spec=session['spec'],doc_empid=session['emp_id'],
                                   appointment=appointment,approve=approve,submitted=submitted, patient_desc=patient_desc,prescribe=prescribe,prescription=prescription)
        else:
          msg="Account does not exists"
          return render_template('appointment.html', msg=msg)
    elif request.method == 'POST':
        msg = 'Please fill out the form !'
    return render_template('appointment.html', msg=msg)


@app.route('/approve/<appId>',methods=['POST','GET'])
def approve(appId):
    if request.method=='POST':
        appId=int(appId)
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('UPDATE book_appointment SET Approved="true" WHERE Appointment_ID=% s',(appId,))

        cursor.execute('SELECT  * FROM book_appointment')
        appointment = cursor.fetchall()

        cursor.execute('SELECT * from book_appointment WHERE Approved="true" ')
        approve = cursor.fetchall()
        cursor.execute('SELECT * from patient_description WHERE  Submitted="true"')
        submitted = cursor.fetchall()
        cursor.execute('SELECT * from prescription WHERE  Prescribed="true"')
        prescribe = cursor.fetchall()
        cursor.execute('SELECT * from patient_description')
        patient_desc = cursor.fetchall()
        cursor.execute('SELECT * from prescription')
        prescription = cursor.fetchall()

        mysql.connection.commit()
        msg="Approved!"

    return render_template('doctor_dashboard.html',success_msg=msg,doc_name=session['doc_name'],spec=session['spec'],doc_empid=session['emp_id'],
                           appointment=appointment,approve=approve,submitted=submitted,prescribe=prescribe,patient_desc=patient_desc,prescription=prescription)

@app.route('/prescribe',methods=['POST','GET'])
def prescribe():
    msg=''
    if request.method=='POST' and  'dis' in request.form and 'dis_desc' in request.form and 'prec' in request.form and 'med' in request.form\
           and 'diet' in request.form and 'work' in request.form:
        dis=request.form['dis']
        dis_desc=request.form['dis_desc']
        prec=request.form['prec']
        med = request.form['med']
        diet = request.form['diet']
        work = request.form['work']
        pid_appid=request.form['pid_appid']
        pid_appid=pid_appid.split(" ")
        pId=pid_appid[0]
        appId=pid_appid[1]

        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute("Select * from prescription where Patient_ID=% s and Appointment_ID=% s", (pId, appId,))
        account = cursor.fetchone()

        cursor.execute('SELECT  * FROM book_appointment')
        appointment = cursor.fetchall()
        pid = []
        for row in appointment:
            pid.append(row['Patient_ID'])

        cursor.execute('SELECT * from book_appointment WHERE Approved="true" ')
        approve = cursor.fetchall()
        cursor.execute('SELECT * from patient_description WHERE  Submitted="true"')
        submitted = cursor.fetchall()
        cursor.execute('SELECT * from prescription WHERE  Prescribed="true"')
        prescribe = cursor.fetchall()

        cursor.execute('SELECT * from patient_description')
        patient_desc = cursor.fetchall()
        cursor.execute('SELECT * from prescription')
        prescription = cursor.fetchall()

        if account:
            return render_template('doctor_dashboard.html', msg=msg, doc_name=session['doc_name'], spec=session['spec'],
                                   doc_empid=session['emp_id'], appointment=appointment, approve=approve,
                                   submitted=submitted, patient_desc=patient_desc, prescribe=prescribe,prescription=prescription)
        else:

          cursor.execute('INSERT INTO prescription VALUES (NULL,% s,% s,% s,% s,% s,% s,% s,% s,% s)',
                            (pId,appId,dis,dis_desc,prec,med,diet,work,'true',))


          mysql.connection.commit()

          msg = "Prescribed!"
        return render_template('doctor_dashboard.html', success_msg=msg, doc_name=session['doc_name'], spec=session['spec'],
                           doc_empid=session['emp_id'],appointment=appointment, approve=approve, submitted=submitted, patient_desc=patient_desc,prescribe=prescribe,prescription=prescription)


@app.route('/admin_register')
def admin_register():
    return render_template('admin_register.html')

@app.route('/func_admin_register',methods=['POST','GET'])
def adminRegister():
    msg=''
    msg1=''
    if request.method=='POST' and 'fname' in request.form and 'empId' in request.form and 'email' in request.form and 'contact' in request.form and 'password' in request.form and 'gender' in request.form:
      fname=request.form['fname']
      empId = request.form['empId']
      email = request.form['email']
      contact = request.form['contact']
      password = request.form['password']
      gender = request.form['gender']
      cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
      cursor.execute('SELECT * FROM admin_register WHERE Employee_ID = % s ', (empId,) )
      account = cursor.fetchone()
      if account:
          msg = 'Account already exists !'
      else:
          cursor.execute('INSERT INTO admin_register VALUES \
      		(NULL, % s, % s, % s, % s, % s,% s)',(fname, empId, email,contact, gender, password,))
          mysql.connection.commit()
          msg1 = 'You have successfully registered !'
          return render_template('admin_register.html',msg=msg1)
    elif request.method == 'POST':
        msg = 'Please fill out the form !'
    return render_template('admin_register.html', msg=msg)


@app.route('/func_admin_login',methods=['POST','GET'])
def adminLogin():
    msg=''
    if request.method=='POST' and 'empId' in request.form and 'password' in request.form:
        empId=request.form['empId']
        password=request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM admin_register WHERE Employee_ID=% s AND Password = % s', (empId,password,))
        account = cursor.fetchone()
        if account:
            session['loggedin'] = True
            session['id'] = account['ID']
            session['name'] = account['FullName']
            msg = 'Logged in successfully !'

            cursor.execute('SELECT * FROM add_doctor ')
            doctor_data = cursor.fetchall()

            cursor.execute('SELECT * FROM patient_register ')
            patient_data = cursor.fetchall()

            cursor.execute('SELECT * FROM book_appointment ')
            appointment = cursor.fetchall()
            cursor.execute('SELECT * FROM prescription ')
            prescription = cursor.fetchall()
            cursor.execute('SELECT * FROM add_doctor where Available=true ')
            available = cursor.fetchall()

            return render_template('admin_dashboard.html', msg=msg, name=session['name'],doctor_data=doctor_data,
                                   patient_data=patient_data,appointment=appointment,prescription=prescription,available=available)
        else:
            msg= 'Incorrect employee id  or password !'
    return render_template('appointment.html', msg=msg)

@app.route('/admin_dashboard')
def adminDashboard():
    return render_template('admin_dashboard.html')


@app.route('/add_doctor',methods=['POST','GET'])
def add_doctor():
    msg=''
    if request.method=='POST' and 'dName' in request.form and 'special' in request.form and 'dEmpId' in request.form and 'demail' in request.form and 'dcontact' in request.form and 'dpassword' in request.form and 'docFees' in request.form:
     dname=request.form['dName']
     spec = request.form['special']
     dEmpId=request.form['dEmpId']
     demail = request.form['demail']
     dcontact = request.form['dcontact']
     dpassword = request.form['dpassword']
     docFees = request.form['docFees']
     cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
     cursor.execute('SELECT * FROM add_doctor WHERE Employee_ID = % s ', (dEmpId,))
     account = cursor.fetchone()
     cursor.execute('SELECT * FROM book_appointment ')
     appointment = cursor.fetchall()
     cursor.execute('SELECT * FROM prescription ')
     prescription = cursor.fetchall()
     cursor.execute('SELECT * FROM add_doctor where Available=true ')
     available = cursor.fetchall()

     if account:
         msg = 'Doctor already exists !'

         cursor.execute('SELECT * FROM add_doctor ')
         doctor_data = cursor.fetchall()

         cursor.execute('SELECT * FROM patient_register ')
         patient_data = cursor.fetchall()

         return render_template('admin_dashboard.html', name=session['name'], error_msg=msg, doctor_data=doctor_data,
                                patient_data=patient_data,appointment=appointment,prescription=prescription,available=available)
     else:
         cursor.execute('INSERT INTO add_doctor VALUES \
           		(NULL, % s, % s, % s, % s, % s,% s,% s)', (dname, spec,dEmpId, demail, dcontact, dpassword,docFees,))
         mysql.connection.commit()
         msg= 'Doctor Added!'
         cursor.execute('SELECT * FROM add_doctor ')
         doctor_data = cursor.fetchall()

         cursor.execute('SELECT * FROM patient_register ')
         patient_data = cursor.fetchall()

         return render_template('admin_dashboard.html',name=session['name'], success_msg=msg,doctor_data=doctor_data,
                                patient_data=patient_data,appointment=appointment,prescription=prescription,available=available)
    elif request.method == 'POST':
        msg = 'Please fill out the form !'
    return render_template('admin_dashboard.html',name=session['name'], error_msg=msg)


@app.route('/delete_doctor',methods=['POST','GET'])
def delete_doctor():
     msg=''
     if request.method=='POST' and 'dEmpId' in request.form:
         dEmpId=request.form['dEmpId']
         cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
         cursor.execute('SELECT * FROM add_doctor WHERE Employee_ID = % s ', (dEmpId,))
         account = cursor.fetchone()
         cursor.execute('SELECT * FROM book_appointment ')
         appointment = cursor.fetchall()
         cursor.execute('SELECT * FROM prescription ')
         prescription = cursor.fetchall()
         cursor.execute('SELECT * FROM add_doctor where Available=true ')
         available = cursor.fetchall()

         if account:
          cursor.execute('DELETE FROM add_doctor WHERE Employee_ID = % s ', (dEmpId,))
          mysql.connection.commit()
          msg="Doctor Deleted!"
          cursor.execute('SELECT * FROM add_doctor ')
          doctor_data = cursor.fetchall()

          cursor.execute('SELECT * FROM patient_register ')
          patient_data = cursor.fetchall()

          return render_template('admin_dashboard.html', success_msg=msg,doctor_data=doctor_data, patient_data=patient_data,
                                 appointment=appointment,prescription=prescription,available=available)
         else:
            msg="No Doctor Exists!"
            cursor.execute('SELECT * FROM add_doctor ')
            doctor_data = cursor.fetchall()

            cursor.execute('SELECT * FROM patient_register ')
            patient_data = cursor.fetchall()
            return render_template('admin_dashboard.html', error_msg=msg,doctor_data=doctor_data, patient_data=patient_data,
                                   appointment=appointment,prescription=prescription,available=available)

     elif request.method == 'POST':
         msg = 'Please fill out the form !'
         return render_template('admin_dashboard.html', error_msg=msg)


@app.route('/logout')
def logout():
  session.pop('loggedin', None)
  session.pop('pid', None)
  session.pop('fname', None)
  return redirect(url_for('appointment'))

@app.route('/doc_logout')
def doc_logout():
  session.pop('loggedin', None)
  cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
  cursor.execute('Update add_doctor SET Available=false WHERE Employee_ID=% s', (session['emp_id'],))
  mysql.connection.commit()
  return redirect(url_for('appointment'))

if __name__ == '__main__':
    app.run(debug=True)