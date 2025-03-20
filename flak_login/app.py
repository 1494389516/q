from flask import Flask, render_template, redirect, url_for, flash, request, send_file
from flask_wtf.csrf import CSRFProtect
from models import db, User
from config import Config
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_talisman import Talisman
from flask_dance.contrib.google import make_google_blueprint, google
from flask_dance.contrib.github import make_github_blueprint, github
import pyotp
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
import io
import random
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)
app.config.from_object(Config)

# 初始化扩展
db.init_app(app)
csrf = CSRFProtect(app)
limiter = Limiter(key_func=get_remote_address)
limiter.init_app(app)

# 配置Flask-Talisman以强制HTTPS
talisman = Talisman(app)

# 配置会话安全防护
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# 登录管理器配置
login_manager = LoginManager(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# 配置Google登录
google_bp = make_google_blueprint(client_id='YOUR_GOOGLE_CLIENT_ID', client_secret='YOUR_GOOGLE_CLIENT_SECRET', redirect_to='google_login')
app.register_blueprint(google_bp, url_prefix='/google_login')

@app.route('/google_login')
def google_login():
    if not google.authorized:
        return redirect(url_for('google.login'))
    resp = google.get('/plus/v1/people/me')
    assert resp.ok, resp.text
    info = resp.json()
    user = User.query.filter_by(email=info['emails'][0]['value']).first()
    if user is None:
        user = User(username=info['displayName'], email=info['emails'][0]['value'])
        db.session.add(user)
        db.session.commit()
    login_user(user)
    return redirect(url_for('home'))

# 配置GitHub登录
github_bp = make_github_blueprint(client_id='YOUR_GITHUB_CLIENT_ID', client_secret='YOUR_GITHUB_CLIENT_SECRET', redirect_to='github_login')
app.register_blueprint(github_bp, url_prefix='/github_login')

@app.route('/github_login')
def github_login():
    if not github.authorized:
        return redirect(url_for('github.login'))
    resp = github.get('/user')
    assert resp.ok, resp.text
    info = resp.json()
    user = User.query.filter_by(email=info['email']).first()
    if user is None:
        user = User(username=info['login'], email=info['email'])
        db.session.add(user)
        db.session.commit()
    login_user(user)
    return redirect(url_for('home'))

# 路由配置
@app.route('/')
def home():
    return render_template('base.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        if User.query.filter_by(username=username).first():
            flash('用户名已存在', 'danger')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('邮箱已被注册', 'danger')
            return redirect(url_for('register'))
        
        new_user = User(username=username, email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('注册成功，请登录', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        remember = request.form.get('remember') == 'on'
        captcha = request.form['captcha']
        
        if captcha != session.get('captcha_text'):
            flash('验证码错误', 'danger')
            return redirect(url_for('login'))
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user, remember=remember)
            if not user.is_authenticated_2fa:
                return redirect(url_for('two_factor_auth'))
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('home'))
        
        flash('用户名或密码错误', 'danger')
    
    return render_template('login.html')

@app.route('/2fa', methods=['GET', 'POST'])
@login_required
def two_factor_auth():
    if request.method == 'POST':
        token = request.form['token']
        if pyotp.TOTP(current_user.otp_secret).verify(token):
            current_user.is_authenticated_2fa = True
            return redirect(url_for('home'))
        else:
            flash('无效的验证码', 'danger')
    
    return render_template('2fa.html')

@app.route('/captcha')
def captcha():
    image = Image.new('RGB', (100, 30), color = (255, 255, 255))
    font = ImageFont.truetype('arial.ttf', 25)
    draw = ImageDraw.Draw(image)
    captcha_text = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=5))
    draw.text((10, 0), captcha_text, font=font, fill=(0, 0, 0))
    
    session['captcha_text'] = captcha_text
    
    buf = io.BytesIO()
    image.save(buf, 'jpeg')
    buf.seek(0)
    return send_file(buf, mimetype='image/jpeg')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)