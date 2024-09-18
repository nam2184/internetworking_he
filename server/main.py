import json
from os import path
import os
import logging

import secrets
from passlib.hash import pbkdf2_sha256


from flask import Flask, Response, request, render_template, redirect, url_for, flash, send_file,abort
from flask_login import LoginManager, login_required, login_user, logout_user, current_user
from flask_wtf import CSRFProtect
from markupsafe import escape
# from flask_apscheduler import APScheduler

from Lib.Python.Forms import AdminChooseUser, AdminModifyUser, AdminGlobalRules, AdminSignup, LoginForm, ModifyUser, SearchForm, SignupApprove, UserSignup, importForm, imageForm
from Lib.Python.ImageHandler import getArtistImages, getImage, getImageID, getTagImages, removeImage, updateImage
from Lib.Python.Users import User, acceptSignups, addSignup, addUser, declineSignups, getUserImages, getUserTags, removeUser, updateUserInfo, updateUserInfoAdmin, verifyUser
from Lib.Python.environmentHandler import createDB, importImage, initialiseSettings, updateEnvironment

from PIL import Image

# Environment Initialisation
# Allow for logging of events
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
csrf = CSRFProtect()
csrf.init_app(app)

# Generate the secret key of the server
if path.exists('secret'):
    with open("secret", 'r') as key:
        app.config['SECRET_KEY'] = key.read()
else:
    with open('secret', 'w') as file:
        file.write(secrets.token_hex(64))
    with open("secret", 'r') as key:
        app.config['SECRET_KEY'] = key.read()

if not path.exists('Images'):
    os.mkdir("Images")


# initialise the login manager
login_manager = LoginManager()
login_manager.init_app(app)

# Create the database if it doesn't exist
if not path.exists('./db/imageSQL.db'):
    createDB()
    

if not path.exists("./db/globalSettings"):
    initialiseSettings()


# If a user attemps to access a page but isn't logged in,
# this redirects them to the login page

@login_manager.unauthorized_handler
def unauthorised():
    return redirect(app.url_for('login'))

# This is used to reload the user object from
# the user ID stored in the session
@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

# This provides functions that need to be accessible in every page
@app.context_processor
def utility_processor():
    # This is used for the search function and passes a list 
    # of all the names of the user's images
    # This allows for autocomplete to work
    
    def searchForm():
        return SearchForm(request.form)
    return dict(searchForm=searchForm)





# END initialisation


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(app.url_for('login'))


@app.route("/_autocomplete", methods=['GET'])
@login_required
def autocomplete():
    imagesRaw = getUserImages(current_user.get_id())
    autocomplete=[]
    for x in imagesRaw:
        # Get image name
        autocomplete.append(x[1])
        

    userTags = getUserTags(current_user.get_id())
    for tag in userTags:
        if tag[0] == None: continue
        autocomplete.append(tag[0])

    return Response(json.dumps(autocomplete), mimetype='application/json')

@app.route('/_tagAutocomplete')
@login_required
def tagAutocomplete():
    tags = []
    userTags = getUserTags(current_user.get_id())
    for tag in userTags:
        if tag[0] == None: continue
        tags.append(tag[0])
    return Response(json.dumps(tags), mimetype='application/json')


# This is the function for the login page
@app.route('/', methods=['GET', 'POST'])
@app.route('/login', methods=['GET', 'POST'])
def login():

    form = LoginForm()

    if form.validate_on_submit():
        login_user(User(form.username.data))

        next = request.args.get('next')
        return app.redirect(next or app.url_for('dashboard'))


    return render_template("login.html", form=form)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = UserSignup()

    if form.validate_on_submit():
        flash('Signed Up - Please wait for an Admin to approve your request')

        addSignup(form)


    return render_template('UserSignup.html', form=form)

# LOGGED IN PAGES

@app.route("/dashboard", methods=['GET', 'POST'])
@login_required
def dashboard(images = None):

    if images == None:
        images = getUserImages(current_user.get_id())
    return render_template('dashboard.html', images = images)

# This is where a user can modify their own settings
@app.route("/settings", methods=['GET', 'POST'])
@login_required
def userSettings():
    form = ModifyUser(obj=current_user)
    if form.validate_on_submit():
        updateUserInfo(current_user.get_id(), form)
        flash("Successfully updated")

    return render_template("userSettings.html", form=form)

# This looks for images in a playlist that are currently missing


@app.route("/image/<ImageID>")
@login_required
def viewImage(ImageID):
    ImageID = escape(ImageID)
    image = getImage(ImageID)
    return render_template('imageView.html', image=image)


@app.route("/download/<filepath>")
@login_required
def downloadImage(filepath):
    filename = escape(filepath)
    filepath = "./Images/" + escape(filepath)
    return send_file(
        filepath,
        download_name=filename,
        as_attachment=True
    )

@app.route("/edit/<imageID>", methods=['GET', 'POST'])
@login_required
def editImage(imageID):
    ImageID = escape(imageID)
    image = getImage(ImageID)
    form = imageForm(userID=current_user.get_id(), obj = image)
    if form.validate_on_submit():
        updateImage(form, current_user.get_id())
        return app.redirect(url_for('viewImage', ImageID=ImageID))
    return render_template('imageEdit.html', form = form, ImageID = ImageID)

# This is used when searching for images to redirect the user to the image page
# This will also filter down images that belong to a certain artist or tag
@app.route("/findImage", methods=['GET','POST'])
@login_required
def findImage():
    searchField = request.form.get("autocomp")
    
    ImageID = getImageID(searchField) 
    # If the search was for a image it will redirect to the image page
    if ImageID != None:
        return redirect(url_for("viewImage", ImageID=ImageID))
    
    # If the search was for an artist, it will instead pass all images belonging to that artist
    artistImages = getArtistImages(searchField, current_user.get_id())
    if artistImages != None:
        return render_template('dashboard.html', images = artistImages)
    # If the search was for an tag, it will instead pass all images belonging to that tag
    tagImages = getTagImages(searchField, current_user.get_id())
    if tagImages != None:
        return render_template('dashboard.html', images = tagImages)


@app.route('/deleteImage/<imageID>')
@login_required
def deleteImage(imageID):
    print(f"Deleting {imageID}")
    removeImage(imageID)
    return redirect(url_for("dashboard"))

# ADMIN Pages

@app.route("/adminSettings")
@login_required
def adminSettings():
    # If someone who is not an admin tries to access this, they will be rejected
    if not current_user.admin:
        abort(403)
    
    return render_template('adminDash.html')


# This allows the admin to select a specific user that will be modified in the following function
@app.route("/adminUserSelect", methods=['GET', 'POST'])
@login_required
def adminUserSelect():
    # If someone who is not an admin tries to access this, they will be rejected
    if not current_user.admin:
        abort(403)

    form = AdminChooseUser()
    if form.validate_on_submit():
        return app.redirect(url_for('adminUserSettings', userID=form.userID.data))

    
    return render_template('adminUserSelect.html', form=form)


# This allows the administrator to modify the settings of a user
@app.route("/adminUser/<userID>", methods=['GET', 'POST'])
@login_required
def adminUserSettings(userID):
    userID = escape(userID)
    # If someone who is not an admin tries to access this, they will be rejected
    if not current_user.admin:
        abort(403)

    user = load_user(userID)
    form = AdminModifyUser(obj=user)
    if form.validate_on_submit():
        if form.update.data:
            updateUserInfoAdmin(form)
            flash("Success")

        elif form.delete.data:
            removeUser(userID)
            flash("User Successfully deleted")

    return render_template('adminUsers.html', form=form, userID=userID)

@app.route("/adminGlobal", methods=['GET', 'POST'])
@login_required
def adminGlobalSettings():
    # If someone who is not an admin tries to access this, they will be rejected
    if not current_user.admin:
        abort(403)
    
    form = AdminGlobalRules()
    if form.validate_on_submit():
        updateEnvironment(form)
        flash("Success")

    return render_template("adminGlobal.html", form=form)


@app.route("/import", methods=['GET', 'POST'])
@login_required
def Import():
    if not current_user.admin:
        abort(403)
    form = importForm()
    if form.validate_on_submit():
        flash("Importing...")
        print(request.files)
        importImage(form, current_user.get_id())
        flash("Success")
        return render_template('importImages.html', form = form)

    return render_template('importImages.html', form = form)

# This is where an admin can create a new user
# This is distinct from the user accessible signup page as it doesn't 
# require an admin to approve the signup i.e the account is immediately created
@app.route("/newUser", methods=['GET', 'POST'])
@login_required
def adminNewUser():
    if not current_user.admin:
        abort(403)

    form = AdminSignup()
    if form.validate_on_submit():
        name = form.username.data
        password = pbkdf2_sha256.hash(form.password.data)
        isAdmin = form.isAdmin.data
        addUser(name, password, isAdmin=isAdmin)
        flash("User added")

    return render_template('adminNewUser.html', form = form)

# This is where an admin can approve a new user
@app.route("/approveUser", methods=['GET', 'POST'])
@login_required
def adminApproveUser():
    if not current_user.admin:
        abort(403)

    form = SignupApprove()
    if form.validate_on_submit():
        if form.approve.data:
            flash ("Users approved")
            acceptSignups(form.signups.data)

        elif form.reject.data:
            flash("Users rejected")
            declineSignups(form.signups.data)

        return redirect(url_for("adminApproveUser"))

    return render_template('adminApprove.html', form = form)

if __name__ == '__main__':

    

    app.run(debug=True, host='0.0.0.0', port=5000)
