// server.js - Node.js Authentication Backend

const express = require('express');
require('dotenv').config(); 
const mysql = require('mysql2/promise'); 
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const bodyParser = require('body-parser');
const cookieParser = require('cookie-parser');

const cors = require('cors');
const path = require('path');

// --- Configuration ---
const app = express();
app.use(cookieParser());
app.use(cors());
const PORT = process.env.PORT || 5000;
const JWT_SECRET = process.env.JWT_SECRET || 'your_jwt_secret_key';
const DB_CONFIG = {
    host: process.env.DB_HOST || 'localhost',
    user: process.env.DB_USER || 'root',
    password: process.env.DB_PASSWORD || 'adityaXXX',
    database: process.env.DB_DATABASE || 'nyay_ai_db',
    waitForConnections: true,
    connectionLimit: 10,
    queueLimit: 0
};

// --- Middleware ---

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(express.static(path.join(__dirname)));

// --- Database Connection Pool ---
let pool;

async function initializeDatabase() {
    try {
        console.log("⚡ Connecting to MySQL...");
        pool = mysql.createPool(DB_CONFIG);
        await pool.query('SELECT 1');
        console.log('✅ MySQL connected successfully.');
    } catch (err) {
        console.error('❌ Database connection failed:', err.message);
        process.exit(1);
    }
}

// --- JWT Middleware ---
const authenticateToken = (req, res, next) => {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1];
    if (!token) return res.status(401).json({ message: 'Authentication token required.' });

    jwt.verify(token, JWT_SECRET, (err, user) => {
        if (err) return res.status(403).json({ message: 'Invalid or expired token.' });
        req.user = user;
        next();
    });
};
app.get('/api/token-info', (req, res) => {
    console.log()
    // Read token from Authorization header
    const authHeader = req.headers['authorization'];
    if (!authHeader) {
        return res.status(401).json({ message: 'No token provided' });
    }
    console.log(authHeader);
    // Header format: "Bearer <token>"
    const token = authHeader.split(' ')[1];
    if (!token) {
        return res.status(401).json({ message: 'Token missing in header' });
    }

    // Verify JWT
    jwt.verify(token, JWT_SECRET, (err, decoded) => {
        if (err) {
            return res.status(403).json({ message: 'Invalid or expired token' });
        }

        // Return full decoded token info
        res.json({
            valid: true,
            issuedAt: new Date(decoded.iat * 1000),
            expiresAt: new Date(decoded.exp * 1000),
            user: decoded.user
        });
    });
});


// --- Routes ---
app.get('/api', (req, res) => {
    res.json({ message: 'Nyay AI Backend API is running!' });
});

/**
 * @route POST /api/signup
 */
app.post('/api/signup', async (req, res) => {
    const { username, email, password } = req.body;
    console.log(req.body);

    if (!username || !email || !password)
        return res.status(400).json({ message: 'All fields are required.' });

    if (password.length < 8)
        return res.status(400).json({ message: 'Password must be at least 8 characters long.' });

    try {
        const [existingUsers] = await pool.query('SELECT id FROM users WHERE email = ?', [email]);
        if (existingUsers.length > 0)
            return res.status(409).json({ message: 'User already exists.' });

        const salt = await bcrypt.genSalt(10);
        const hashedPassword = await bcrypt.hash(password, salt);

        const [result] = await pool.query(
            'INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
            [username, email, hashedPassword]
        );

        // ✅ Create JWT token with username and email
        const payload = { user: { id: result.insertId, username, email } };
        const token = jwt.sign(payload, JWT_SECRET, { expiresIn: '1h' });

        res.status(201).json({
            message: 'User registered successfully!',
            token,
            user: { id: result.insertId, username, email }
        });
    } catch (err) {
        console.error('❌ Error during signup:', err.message);
        res.status(500).json({ message: 'Server error during signup.' });
    }
});

/**
 * @route POST /api/login
 */
app.post('/api/login', async (req, res) => {
    const { email, password } = req.body;
    if (!email || !password)
        return res.status(400).json({ message: 'Email and password are required.' });

    try {
        const [users] = await pool.query('SELECT id, username, email, password FROM users WHERE email = ?', [email]);
        if (users.length === 0)
            return res.status(401).json({ message: 'Invalid credentials.' });

        const user = users[0];
        const isMatch = await bcrypt.compare(password, user.password);
        if (!isMatch)
            return res.status(401).json({ message: 'Invalid credentials.' });

        // ✅ Create JWT token with username and email
        const payload = { user: { id: user.id, username: user.username, email: user.email } };
        const token = jwt.sign(payload, JWT_SECRET, { expiresIn: '1h' });

        res.json({
            message: 'Login successful!',
            token,
            user: { id: user.id, username: user.username, email: user.email }
        });
    } catch (err) {
        console.error('❌ Error during login:', err.message);
        res.status(500).json({ message: 'Server error during login.' });
    }
});

/**
 * @route GET /api/protected
 */
app.get('/api/protected', authenticateToken, (req, res) => {
    res.json({
        message: `Welcome, ${req.user.user.username}!`,
        userData: req.user.user
    });
});

// ✅ Serve frontend
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// --- Start Server ---
async function startServer() {
    await initializeDatabase();
    app.listen(PORT, () => {
        console.log(`🚀 Server running on port ${PORT}`);
    });
}

startServer();
