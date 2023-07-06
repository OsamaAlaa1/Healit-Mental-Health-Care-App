
# import needed packajes 
import mysql.connector 


# function to connect Database:
def connect_excute_db(db_name,query):

    # Establish a connection to the database
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="122333",
        database= db_name 
    )


    # Create a cursor object to interact with the database
    cursor = db.cursor()

    # Execute a query
    cursor.execute(query)

    # Fetch the results
    result = cursor.fetchall()

        
    # Close the cursor and database connections
    cursor.close()
    db.close()
    
    #return the results
    return result


def register(user_name, email, password, user_type):
    try:
        # Establish a connection to the database
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="122333",
            database='mentalhealth_db'
        )

        # Create a cursor object to execute SQL queries
        cursor = db.cursor()

        # Define the SQL query to insert values into the table
        query = "INSERT INTO users (user_name, email, password, user_type) VALUES (%s, %s, %s, %s)"

        # Execute the SQL query with the provided values
        cursor.execute(query, (user_name, email, password, user_type))

        # Commit the changes to the database
        db.commit()

        # Close the cursor and the database connection
        cursor.close()
        db.close()
        return ("Account Created Successfully!")
    
    except mysql.connector.Error as error:
        return(f"Error connecting to MySQL: {error}")

    except Exception as e:
        return(f"An error occurred: {e}")


def login(username, password):
    try:
        # Establish a connection to the database
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="122333",
            database='mentalhealth_db'
        )

        # Create a cursor object to execute SQL queries
        cursor = db.cursor()

        # Define the SQL query to select values from the table
        query = "SELECT * FROM users WHERE user_name = %s AND password = %s"

        # Execute the SQL query with the provided values
        cursor.execute(query, (username, password))

        # Fetch the result
        result = cursor.fetchone()

        # Close the cursor and the database connection
        cursor.close()
        db.close()

        if result:
            return "Logged in Successfully !"
        else:
            return "Error: Invalid username or password, Don't have an account? try to Register."

    except mysql.connector.Error as error:
        return f"Error connecting to MySQL: {error}"

    except Exception as e:
        return f"An error occurred: {e}"
