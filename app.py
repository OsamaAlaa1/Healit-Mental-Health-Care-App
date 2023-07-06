
from flask import Flask, jsonify,render_template,request,redirect,url_for,flash

# import database function
import models.database as db

# import model
from models.model import excute_detection_model

# import get response function from chatbot file 
from chatbot_files.chat import get_response

# app initialization 
app = Flask(__name__)

log = False
username = 'Login'
negative = 'no'
depressed = 'no'
red_flag = 'no'


"""Rander pages Section"""


# render home page
@app.route('/')
@app.route('/home')
@app.route('/index.html')
def home():
    return render_template('home.html')

@app.get('/chatbot')
def base():
    return render_template('base.html')

@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    # check if text is valid
    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)


# render library page
@app.route('/library',methods = ['POST','GET'])
def library():

    if request.method == 'POST': 
        
        #get disorder name 
        disorder = str(request.form['search_text']).capitalize()
        
        #connect and excute queries to get general Information 
    
        disorder_id = db.connect_excute_db("mentalhealth_db",f"SELECT disorder_id FROM disorders WHERE disorder_name = '{disorder}'")[0][0]    # as it's list of tuples
        general_info = db.connect_excute_db("mentalhealth_db",f"SELECT disorder_info FROM disorders WHERE disorder_name = '{disorder}'")[0][0]  # as it's list of tuples
        risk_factors = db.connect_excute_db("mentalhealth_db",f"SELECT C.cause FROM disorder_cause AS DC, causes AS C WHERE DC.disorder_id = {disorder_id} AND C.cause_id = DC.cause_id;")
        symptoms = db.connect_excute_db("mentalhealth_db",f"SELECT S.symptom FROM disorder_symptom AS DS, symptoms AS S WHERE DS.disorder_id = {disorder_id} AND S.symptom_id = DS.symptom_id;")
        treatments = db.connect_excute_db("mentalhealth_db",f"SELECT T.treatment FROM disorder_treatment AS DT, treatments AS T WHERE DT.disorder_id = {disorder_id} AND T.treatment_id = DT.treatment_id;")

        return render_template('library.html',disorder= disorder, general_info = general_info, risk_factors = risk_factors, symptoms = symptoms, treatments = treatments, enumerate = enumerate, len = len)

    return render_template( 'library.html',disorder ="Mental Disorders", general_info ="""
    Mental disorders, also called mental illnesses, refer to a wide range of mental health conditions that affect an individual's mood, behavior, 
    and thinking. These disorders can impact a person's ability to function normally in their daily life and can affect anyone regardless of age, gender, race, or socio-economic status.""", 
    risk_factors = [['Genetics and family history'],['Life events'],['Poverty and financial stress'],['physical health conditions']], 
    symptoms= [['Feelings of sadness'],['Loss of interest'],['Changes in appetite'],['Lack of energy']], 
    treatments=[['Medications'],['Psychotherapy'],['Hospitalization'],['Self-care']],
    enumerate=enumerate , len=len)




# render Register page
@app.route('/register', methods = ['GET','POST'])
def register():
    
    if request.method == 'POST':

        username = request.form['username']
        email = request.form['email']
        password = request.form['pass']
        user_type = request.form['user_type']
        
        message = db.register(username,email,password,user_type)

        global log
        log = True

        return render_template('register.html' ,message = message)


    return render_template('register.html' , message = '')




@app.route('/login', methods = ['GET','POST'])
def login():
    
    if request.method == 'POST':

        username = request.form['username']
        password = request.form['pass']
        
        db.login(username,password)
        
        global log 
        log = True


        return render_template('detector.html',username = username)


    return render_template('login.html')


# render detector page
@app.route('/detector',methods = ['GET','POST'])
def detector():

    if not log:
        return redirect(url_for('login'))  # Redirect to the login page if not logged in


    if request.method == 'POST':

        gender = request.form['gender'] 
        answers = []
        text_answer = ""

        for question in range(1,11):
            answers.append(int(request.form[f'question-{question}']))
            
        text_answer = request.form['speech-to-text']
        text_answer += ' ' + request.form['text-answer']

        sentiment, depression, red_flags = excute_detection_model(answers,text_answer)
        
        return render_template('detector.html',sentiment=sentiment, depression=depression, red_flags=red_flags)
    
    return render_template('detector.html',sentiment="No Results yet", depression = "No Results yet", red_flags="No Results yet")


# render treatment page
@app.route('/treatment')
def treatment():
    return render_template('treatment.html')

# render treatment page
@app.route('/doctor')
def doctor():
    return render_template('doctor.html')


# render about page
@app.route('/about')
def about():
    return render_template('about.html')

# render model page
@app.route('/model')
def model():
    return render_template('model.html')

# render model page
@app.route('/support')
def support():
    return render_template('support.html')



# run application
if __name__ == "__main__":
    app.run(debug=False,port=8000)
