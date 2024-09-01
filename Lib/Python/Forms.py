import sqlite3 
from wsgiref.validate import validator
from flask_wtf import FlaskForm
from flask_wtf.form import _Auto
from wtforms import (StringField, BooleanField, PasswordField, SelectField, 
                     SelectMultipleField, SubmitField)
from wtforms.validators import InputRequired, Length, ValidationError
from wtforms.widgets import PasswordInput, CheckboxInput, ListWidget

from passlib.hash import pbkdf2_sha256

from os import path
from Lib.Python.ImageHandler import getAllImages, getImageTag
from Lib.Python.Users import getAllUsers, getSignups

from Lib.Python.environmentHandler import getImportDirectory, getImportImages

def validate_password():
    message = 'Username or password is incorrect'

    def _sqlGetHash(username):
        conn = sqlite3.connect('./db/imageSQL.db')
        return conn.execute('SELECT password FROM Users where UserName = ?',
                            (username,)).fetchone()

    def _validatepassword(form, field):

        passwordHash = _sqlGetHash(form.username.data)
        if passwordHash == None:
            raise (ValidationError(message))

        if not pbkdf2_sha256.verify(form.password.data, passwordHash[0]):
            raise (ValidationError(message))

    return _validatepassword


def validate_username():
    def _sqlGetUsername(username):
        conn = sqlite3.connect('./db/imageSQL.db')
        return conn.execute('SELECT UserId FROM Users where UserName = ?',
                            (username,)).fetchone()

    def _sqlGetID(id):
        conn = sqlite3.connect('./db/imageSQL.db')
        return conn.execute('SELECT UserName FROM Users where UserId = ?',
                            (id,)).fetchone()

    def _checkExists(form, field):
        # Checks if the username exists
        x = _sqlGetUsername(field.data)
        if x == None:
            return

        # we know the username is being used
        # here we check if it is being used by the user requesting a change
        # This is the case when updating the playlist ID but not the username
        elif x[0] == form.id.data:
            return

        else:
            raise (ValidationError("Username is already being used"))

    return _checkExists


def verifyFilePath():

    def _checkExists(form, field):
        if not path.exists(field.data):
            raise (ValidationError("This path does not exist"))
        
        if not path.isdir(field.data):
            raise(ValidationError("This is not a folder (most likely a file)"))
        
        if field.data == "Images":
            raise(ValidationError("This is a forbidden folder"))
        
        return

    return _checkExists


class LoginForm(FlaskForm):
    username = StringField("User Name", validators=[InputRequired()])
    password = PasswordField("Password", validators=[
                             InputRequired(), Length(min=3), validate_password()])


class UserSignup(FlaskForm):
    username = StringField("User Name", validators=[
                           InputRequired(), validate_username()])
    
    password = PasswordField("Password", validators=[
                             InputRequired(), Length(min=3)])
    

class AdminSignup(UserSignup):
    isAdmin = BooleanField("isAdmin")

class SignupApprove(FlaskForm):
    signups = SelectMultipleField("Users", 
                                widget=ListWidget(prefix_label=False),
                                option_widget=CheckboxInput())
    
    approve = SubmitField()
    reject = SubmitField()

    def __init__(self, *args, **kwargs):
        super(SignupApprove, self).__init__(*args, **kwargs)
        self.signups.choices= getSignups()


class ModifyUser(FlaskForm):
    username = StringField("User Name", validators=[
                           InputRequired(), validate_username()])
    playListID = StringField("Playlist ID", )
    id = StringField()


class AdminModifyUser(ModifyUser):
    admin = BooleanField("isAdmin")
    password = StringField("Password", widget=PasswordInput(hide_value=False), validators=[InputRequired(), Length(min=3)])

    update = SubmitField()
    delete = SubmitField()

class AdminGlobalRules(FlaskForm):
    importDirectory = StringField("Import Folder Path", validators=[Length(min=2), verifyFilePath()])
    def __init__(self, *args, **kwargs):
        super(AdminGlobalRules, self).__init__(*args, **kwargs)
        if getImportDirectory()!="NONE":
            self.importDirectory.data = getImportDirectory()

class AdminChooseUser(FlaskForm):
    userID = SelectField("User to edit")
    def __init__(self, *args, **kwargs):
        super(AdminChooseUser, self).__init__(*args, **kwargs)

        self.userID.choices = getAllUsers()



class imageForm(FlaskForm):
    name = StringField("Image Name", validators=[
                           InputRequired()])
    
    id= StringField()
    tag= StringField("Image Tags", id='search_tagAutocomplete')


    def __init__(self, userID, *args, **kwargs):
        super(imageForm, self).__init__(*args, **kwargs)
        if self.tag.data == None:
            self.tag.data = getImageTag(self.id.data, userID)
        

# Will most likely need to be modified 
class importForm(FlaskForm):
    importFile = SelectField("File to be Imported")
    image = SelectField("Image")

    def __init__(self, *args, **kwargs):
        super(importForm, self).__init__(*args, **kwargs)
        
        self.importFile.choices = getImportImages()
        sortedImages = getAllImages()
        sortedImages = sorted(sortedImages, key=lambda x: x[1])
        self.image.choices= sortedImages

class SearchForm(FlaskForm):
    autocomp = StringField('Search Image', id='search_autocomplete')

