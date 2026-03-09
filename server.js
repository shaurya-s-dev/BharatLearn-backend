/**
 * ╔══════════════════════════════════════════════════════════════════╗
 * ║        BharatLearn Dev Coach  –  Backend API Server  v5         ║
 * ║        Single-file Express server (server.js)                   ║
 * ╠══════════════════════════════════════════════════════════════════╣
 * ║  AI Provider: HuggingFace Inference API                         ║
 * ║  Routes:                                                         ║
 * ║    GET  /health                   liveness probe                 ║
 * ║    GET  /auth/google              start OAuth flow               ║
 * ║    GET  /auth/google/callback     OAuth callback                 ║
 * ║    GET  /auth/logout              destroy session                ║
 * ║    GET  /auth/me                  current user (or null)         ║
 * ║    POST /api/chat                 general AI chat (saves to DB)  ║
 * ║    POST /api/syllabus             syllabus → 24-week plan        ║
 * ║    POST /api/quiz                 topic → 10 quiz questions      ║
 * ║    POST /api/viva                 code file → 20 viva questions  ║
 * ║    POST /api/debug                code → hints (no solutions)    ║
 * ║    POST /api/upload               upload file to S3              ║
 * ║    GET  /api/progress/:userId     get learning progress          ║
 * ║    POST /api/progress             save learning progress         ║
 * ╚══════════════════════════════════════════════════════════════════╝
 */

import "dotenv/config";
import express      from "express";
import helmet       from "helmet";
import cors         from "cors";
import session      from "express-session";
import passport     from "passport";
import { Strategy as GoogleStrategy } from "passport-google-oauth20";
import rateLimit    from "express-rate-limit";
import morgan       from "morgan";
import Joi          from "joi";
import multer       from "multer";
import { createRequire } from "module";
import { v4 as uuid } from "uuid";
import jwt from "jsonwebtoken";

// ── CHANGED: HuggingFace replaces GoogleGenerativeAI ──────────────────────────
import Groq from "groq-sdk";

import { DynamoDBClient }                                                       from "@aws-sdk/client-dynamodb";
import { DynamoDBDocumentClient, PutCommand, GetCommand, UpdateCommand }        from "@aws-sdk/lib-dynamodb";
import { S3Client, PutObjectCommand, GetObjectCommand }                         from "@aws-sdk/client-s3";
import { getSignedUrl }                                                         from "@aws-sdk/s3-request-presigner";
import { CloudWatchLogsClient, PutLogEventsCommand, CreateLogStreamCommand,
         DescribeLogStreamsCommand }                                             from "@aws-sdk/client-cloudwatch-logs";

const require = createRequire(import.meta.url);

// ─── Config ───────────────────────────────────────────────────────────────────
const PORT                 = process.env.PORT                || 3000;
const IS_PROD              = process.env.NODE_ENV === "production";
const FRONTEND_URL         = process.env.FRONTEND_URL        || "http://localhost:3001";
const SESSION_SECRET       = process.env.SESSION_SECRET      || "dev-only-secret-change-in-prod";
const JWT_SECRET           = process.env.JWT_SECRET          || SESSION_SECRET;
const GOOGLE_CLIENT_ID     = process.env.GOOGLE_CLIENT_ID    || "";
const GOOGLE_CLIENT_SECRET = process.env.GOOGLE_CLIENT_SECRET || "";
const AWS_REGION           = process.env.AWS_REGION          || "us-east-1";
const S3_BUCKET            = process.env.S3_BUCKET           || "bharatlearn-storage";
const CW_LOG_GROUP         = process.env.CW_LOG_GROUP        || "/bharatlearn/api";
const CW_LOG_STREAM        = `server-${new Date().toISOString().slice(0, 10)}`;

const GROQ_MODEL = process.env.GROQ_MODEL || "llama-3.1-8b-instant";
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY || "" });

// ─── AWS Clients ──────────────────────────────────────────────────────────────
const awsCredentials = process.env.AWS_ACCESS_KEY_ID
  ? { accessKeyId: process.env.AWS_ACCESS_KEY_ID, secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY }
  : undefined;
const awsConfig = { region: AWS_REGION, ...(awsCredentials && { credentials: awsCredentials }) };

const dynamo  = DynamoDBDocumentClient.from(new DynamoDBClient(awsConfig), {
  marshallOptions: { removeUndefinedValues: true },
});
const s3      = new S3Client(awsConfig);
const cwLogs  = new CloudWatchLogsClient(awsConfig);

// ─── CloudWatch Structured Logger ────────────────────────────────────────────
let _cwSequenceToken = null;
let _cwBuffer        = [];
let _cwFlushTimer    = null;

async function _ensureLogStream() {
  try {
    await cwLogs.send(new CreateLogStreamCommand({ logGroupName: CW_LOG_GROUP, logStreamName: CW_LOG_STREAM }));
  } catch (e) {
    if (e.name !== "ResourceAlreadyExistsException") throw e;
    const res = await cwLogs.send(new DescribeLogStreamsCommand({
      logGroupName: CW_LOG_GROUP, logStreamNamePrefix: CW_LOG_STREAM,
    }));
    _cwSequenceToken = res.logStreams?.[0]?.uploadSequenceToken ?? null;
  }
}

async function _flushCWLogs() {
  if (!_cwBuffer.length) return;
  const events = _cwBuffer.splice(0);
  try {
    const res = await cwLogs.send(new PutLogEventsCommand({
      logGroupName: CW_LOG_GROUP, logStreamName: CW_LOG_STREAM,
      logEvents: events,
      ...(_cwSequenceToken && { sequenceToken: _cwSequenceToken }),
    }));
    _cwSequenceToken = res.nextSequenceToken;
  } catch (e) {
    console.error("[CW] Flush failed:", e.message);
  }
}

function log(level, event, data = {}) {
  const entry = { ts: new Date().toISOString(), level, event, ...data };
  const fn = level === "ERROR" ? console.error : level === "WARN" ? console.warn : console.log;
  fn(`[${level}] ${event}`, Object.keys(data).length ? data : "");

  if (process.env.AWS_ACCESS_KEY_ID) {
    _cwBuffer.push({ timestamp: Date.now(), message: JSON.stringify(entry) });
    if (!_cwFlushTimer) _cwFlushTimer = setInterval(_flushCWLogs, 2000);
  }
}

// ─── DynamoDB Tables ──────────────────────────────────────────────────────────
const TABLES = {
  USERS:    "bharatlearn-users",
  PROGRESS: "bharatlearn-progress",
  SESSIONS: "bharatlearn-sessions",
  PLANS:    "bharatlearn-learning-plans",
};

async function upsertUser(profile) {
  const user = {
    userId: profile.id, email: profile.emails?.[0]?.value ?? "",
    name: profile.displayName, avatar: profile.photos?.[0]?.value ?? "",
    createdAt: new Date().toISOString(), updatedAt: new Date().toISOString(),
  };
  await dynamo.send(new PutCommand({ TableName: TABLES.USERS, Item: user }));
  log("INFO", "user_login", { userId: user.userId, email: user.email });
  return user;
}

async function saveChatSession(userId, sessionId, chatHistory) {
  await dynamo.send(new PutCommand({
    TableName: TABLES.SESSIONS,
    Item: { sessionId, userId, chatHistory, timestamp: new Date().toISOString() },
  }));
}

async function getProgress(userId) {
  const res = await dynamo.send(new GetCommand({ TableName: TABLES.PROGRESS, Key: { userId } }));
  return res.Item ?? null;
}

async function saveProgress(userId, topic, data) {
  await dynamo.send(new UpdateCommand({
    TableName: TABLES.PROGRESS, Key: { userId },
    UpdateExpression: "SET #topic = :data, updatedAt = :ts",
    ExpressionAttributeNames: { "#topic": topic },
    ExpressionAttributeValues: { ":data": data, ":ts": new Date().toISOString() },
  }));
}

async function saveLearningPlan(userId, plan) {
  const planId = uuid();
  await dynamo.send(new PutCommand({
    TableName: TABLES.PLANS,
    Item: { planId, userId, plan, createdAt: new Date().toISOString() },
  }));
  return planId;
}

// ─── S3 Helpers ───────────────────────────────────────────────────────────────
function getMimeType(ext) {
  const map = { pdf: "application/pdf", txt: "text/plain", py: "text/x-python",
                js: "application/javascript", ts: "text/typescript",
                java: "text/x-java-source", cpp: "text/x-c++src" };
  return map[ext] ?? "application/octet-stream";
}

async function uploadToS3(buffer, originalName, userId, folder = "uploads") {
  const ext = originalName.split(".").pop().toLowerCase();
  const key = `${folder}/${userId}/${uuid()}.${ext}`;
  await s3.send(new PutObjectCommand({
    Bucket: S3_BUCKET, Key: key, Body: buffer,
    ContentType: getMimeType(ext), Metadata: { userId, originalName },
  }));
  log("INFO", "s3_upload", { key, userId, folder, bytes: buffer.length });
  return key;
}

async function getPresignedUrl(key) {
  return getSignedUrl(s3, new GetObjectCommand({ Bucket: S3_BUCKET, Key: key }), { expiresIn: 3600 });
}

// ─── CHANGED: HuggingFace AI Helper (replaces callBedrock/Gemini) ─────────────
//
//  Uses HuggingFace's OpenAI-compatible chatCompletion endpoint.
//  Works with any instruction-tuned model on HF (Mistral, Llama, Zephyr, etc.)
//  The function signature is intentionally identical to the old callBedrock
//  so NO other code in the file needs to change.
//
async function callBedrock(system, userContent, maxTokens = 2048) {
  const messages = Array.isArray(userContent)
    ? userContent : [{ role: "user", content: userContent }];

  const t0 = Date.now();
  const response = await groq.chat.completions.create({
    model: GROQ_MODEL,
    messages: [
      { role: "system", content: system },
      ...messages.map(m => ({ role: m.role, content: m.content })),
    ],
    max_tokens: maxTokens,
    temperature: 0.7,
  });

  const text = response.choices?.[0]?.message?.content?.trim();
  if (!text) throw appError(503, "AI model returned empty response");
  log("INFO", "groq_call", { ms: Date.now() - t0, model: GROQ_MODEL });
  return text;
}

function parseAIJson(raw) {
  const clean = raw.replace(/```json[\s\S]*?```|```[\s\S]*?```/g, s => s.replace(/```json|```/gi, "").trim()).trim();
  try   { return JSON.parse(clean); }
  catch { throw appError(503, "AI returned malformed JSON – please retry"); }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
function appError(status, message, code = "APP_ERROR") {
  const e = new Error(message); e.status = status; e.code = code; e.isOp = true; return e;
}
function validate(schema, data) {
  const { error, value } = schema.validate(data, { abortEarly: false, stripUnknown: true });
  if (error) throw appError(400, error.details.map(d => d.message).join("; "), "VALIDATION");
  return value;
}

// ─── Express Setup ────────────────────────────────────────────────────────────
const app = express();
app.use(helmet({ contentSecurityPolicy: IS_PROD ? undefined : false }));
app.use(cors({
  origin: function(origin, callback) {
    const allowed = [
      "http://localhost:3001",
      "http://localhost:3000",
    ];
    // Allow any vercel.app URL from your account
    if (!origin || allowed.includes(origin) || origin.endsWith(".vercel.app")) {
      callback(null, true);
    } else {
      callback(new Error("Not allowed by CORS"));
    }
  },
  methods: ["GET","POST","PUT","DELETE","OPTIONS"],
  credentials: true,
}));
app.use(express.urlencoded({ extended: false, limit: "32kb" }));
app.use(express.json({ limit: "32kb" }));
app.disable("x-powered-by");
app.use(morgan(IS_PROD ? "combined" : "dev"));
app.use((req, _res, next) => { req.id = uuid(); next(); });
app.use(passport.initialize());

app.get("/", (req, res) => {
  res.json({
    message: "BharatLearn backend is running 🚀",
    service: "AI Dev Coach API"
  });
});

app.get("/health", (req, res) => {
  res.json({ status: "ok" });
});

// ─── Google OAuth (JWT-based, no sessions) ────────────────────────────────────
if (GOOGLE_CLIENT_ID && GOOGLE_CLIENT_SECRET) {
  passport.use(new GoogleStrategy(
    { clientID: GOOGLE_CLIENT_ID, clientSecret: GOOGLE_CLIENT_SECRET,
      callbackURL: IS_PROD
        ? "https://bharat-learn-frontend.vercel.app/auth/google/callback"
        : `http://localhost:${PORT}/auth/google/callback`,
    },
    async (_at, _rt, profile, done) => {
      try { done(null, await upsertUser(profile)); }
      catch (err) {
        log("WARN", "auth_db_fallback", { error: err.message });
        done(null, { userId: profile.id, name: profile.displayName,
                     email: profile.emails?.[0]?.value, avatar: profile.photos?.[0]?.value });
      }
    }
  ));
}
passport.serializeUser((u, done)   => done(null, u));
passport.deserializeUser((u, done) => done(null, u));

// ─── Rate Limiters ────────────────────────────────────────────────────────────
const rl = (windowMs, max, msg) => rateLimit({
  windowMs, max, standardHeaders: true, legacyHeaders: false,
  message: { success: false, error: { code: "RATE_LIMIT", message: msg } },
});
app.use("/api/syllabus", rl(60_000, 6,  "AI limit: 6/min"));
app.use("/api/quiz",     rl(60_000, 8,  "AI limit: 8/min"));
app.use("/api/viva",     rl(60_000, 6,  "AI limit: 6/min"));
app.use("/api/debug",    rl(60_000, 10, "AI limit: 10/min"));
app.use("/api/chat",     rl(60_000, 15, "Chat limit: 15/min"));
app.use("/auth/",        rl(15 * 60_000, 20, "Too many auth requests. Try in 15 min."));
app.use("/api/",         rl(60_000, 60, "API limit reached"));

// ─── Multer ───────────────────────────────────────────────────────────────────
const upload = multer({
  storage: multer.memoryStorage(),
  limits:  { fileSize: 10 * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    const ext  = "." + file.originalname.split(".").pop().toLowerCase();
    const good = [".pdf",".txt",".py",".js",".ts",".java",".cpp",".c",".cs",".go",".rb",".php"];
    cb(good.includes(ext) ? null : appError(400, `Unsupported file type: ${ext}`), good.includes(ext));
  },
});

// ─── Routes ───────────────────────────────────────────────────────────────────

app.get("/health", (_req, res) => {
  log("INFO", "health_check");
  res.json({ ok: true, env: process.env.NODE_ENV, ts: new Date().toISOString() });
});

app.get("/auth/google", passport.authenticate("google", { scope: ["profile","email"], session: false }));

app.get("/auth/google/callback",
  passport.authenticate("google", { failureRedirect: `${FRONTEND_URL}/settings?error=auth_failed`, session: false }),
  (req, res) => {
    const token = jwt.sign(
      { userId: req.user.userId, name: req.user.name, email: req.user.email, avatar: req.user.avatar },
      JWT_SECRET,
      { expiresIn: "7d" }
    );
    log("INFO", "jwt_issued", { userId: req.user.userId });
    res.redirect(`${FRONTEND_URL}/dashboard?token=${token}`);
  }
);

app.get("/auth/logout", (req, res) => {
  res.json({ success: true, message: "Logged out" });
});

app.get("/auth/me", (req, res) => {
  const auth = req.headers.authorization;
  if (!auth?.startsWith("Bearer ")) return res.json({ success: true, user: null });
  try {
    const user = jwt.verify(auth.slice(7), JWT_SECRET);
    res.json({ success: true, user });
  } catch {
    res.json({ success: true, user: null });
  }
});

app.post("/api/upload", upload.single("file"), async (req, res, next) => {
  try {
    if (!req.file) throw appError(400, "No file provided");
    const userId = req.user?.id ?? "anonymous";
    const key    = await uploadToS3(req.file.buffer, req.file.originalname, userId, req.body.folder ?? "uploads");
    const url    = await getPresignedUrl(key);
    res.json({ success: true, data: { key, url, name: req.file.originalname, size: req.file.size } });
  } catch (e) { next(e); }
});

app.get("/api/progress/:userId", async (req, res, next) => {
  try { res.json({ success: true, data: await getProgress(req.params.userId) }); }
  catch (e) { next(e); }
});

const progressSchema = Joi.object({
  userId: Joi.string().required(),
  topic:  Joi.string().trim().min(1).max(200).required(),
  data:   Joi.object().required(),
});

app.post("/api/progress", async (req, res, next) => {
  try {
    const { userId, topic, data } = validate(progressSchema, req.body);
    await saveProgress(userId, topic, data);
    log("INFO", "progress_saved", { userId, topic });
    res.json({ success: true, message: "Progress saved" });
  } catch (e) { next(e); }
});

const chatSchema = Joi.object({
  message:      Joi.string().trim().min(1).max(4000).required(),
  history:      Joi.array().items(
    Joi.object({ role: Joi.string().valid("user","assistant").required(), content: Joi.string().max(4000).required() })
  ).max(20).default([]),
  systemPrompt: Joi.string().trim().max(1000).optional(),
  sessionId:    Joi.string().optional(),
});

app.post("/api/chat", async (req, res, next) => {
  try {
    const { message, history, systemPrompt, sessionId } = validate(chatSchema, req.body);
    const system   = systemPrompt || `You are Astra, the AI tutor on BharatLearn Dev Coach — an Indian CS learning platform.

Personality: warm, encouraging, concise. Use simple English a college student would understand.
Teaching style: Socratic — guide with questions before giving direct answers. Help students think, not just copy.

Rules:
- NEVER write complete homework or assignment solutions. Give hints, partial examples, and explain concepts instead.
- If the question is off-topic (not CS, programming, or academics), politely redirect: "I'm best at CS topics — want to try a coding question?"
- Format code blocks with the correct language tag (e.g. \`\`\`python).
- Keep responses under 400 words unless the student explicitly asks for more detail.
- If you are unsure or don't know, say so honestly — never fabricate facts.
- When explaining errors, describe WHY something is wrong, not just what to change.
- Use analogies and real-world examples when explaining abstract concepts.`;
    const messages = [...history, { role: "user", content: message }];
    const reply    = await callBedrock(system, messages, 1024);
    log("INFO", "chat_message", { userId: req.user?.id ?? "anon", msgLen: message.length });
    if (req.user?.id) {
      saveChatSession(req.user.id, sessionId ?? uuid(), [...messages, { role: "assistant", content: reply }])
        .catch(e => log("WARN", "chat_save_failed", { error: e.message }));
    }
    res.json({ success: true, data: { reply } });
  } catch (e) { next(e); }
});

const syllabusBodySchema = Joi.object({ text: Joi.string().trim().min(20).max(8000).required() });

app.post("/api/syllabus", upload.single("file"), async (req, res, next) => {
  try {
    let text = "", s3Key = null;
    if (req.file) {
      const userId = req.user?.id ?? "anonymous";
      s3Key = await uploadToS3(req.file.buffer, req.file.originalname, userId, "syllabus")
        .catch(e => { log("WARN", "syllabus_s3_failed", { error: e.message }); return null; });
      text = req.file.mimetype === "application/pdf"
        ? (await require("pdf-parse")(req.file.buffer).catch(() => { throw appError(400, "PDF parse failed – try pasting text instead"); })).text?.trim() || ""
        : req.file.buffer.toString("utf-8");
    } else {
      ({ text } = validate(syllabusBodySchema, req.body));
    }
    if (text.length < 20) throw appError(400, "Syllabus content too short");
    log("INFO", "syllabus_generating", { userId: req.user?.id ?? "anon", textLen: text.length });

    const raw  = await callBedrock(
      `You are an expert CS curriculum designer familiar with Indian university syllabi (AKTU, VTU, Mumbai University, etc.).

Task: Convert the given syllabus into a structured 24-week learning plan.

Respond ONLY with valid JSON (no markdown, no fences, no explanation outside the JSON).
Schema: {"title":string,"description":string,"totalWeeks":24,"weeks":[{"week":number,"theme":string,"topics":string[],"tasks":string[],"milestone":string}]}

Rules:
- Exactly 24 weeks, no more, no less.
- 2-4 topics per week, 1-2 practical tasks, one milestone per week.
- Follow a progressive difficulty curve: foundational concepts in weeks 1-8, intermediate in 9-16, advanced + revision in 17-24.
- Include at least 2 dedicated revision/practice weeks (e.g., week 12 and week 23).
- Tasks should be hands-on (write code, build a mini-project, solve problems) — not just "read chapter X".
- Milestones should be concrete and verifiable (e.g., "Can implement a linked list from scratch").
- If the syllabus is vague, infer reasonable CS topics to fill gaps.`,
      `Syllabus:\n${text.slice(0, 6000)}\n\nGenerate the 24-week JSON plan now.`,
      4096
    );
    const plan   = parseAIJson(raw);
    const planId = req.user?.id
      ? await saveLearningPlan(req.user.id, plan).catch(e => { log("WARN", "plan_save_failed", { error: e.message }); return null; })
      : null;
    log("INFO", "syllabus_generated", { userId: req.user?.id ?? "anon", planId });
    res.json({ success: true, data: { plan, planId, s3Key } });
  } catch (e) { next(e); }
});

const quizSchema = Joi.object({
  topic:      Joi.string().trim().min(2).max(200).required(),
  language:   Joi.string().trim().max(50).default("Python"),
  difficulty: Joi.string().valid("Beginner","Intermediate","Advanced").default("Intermediate"),
});

app.post("/api/quiz", async (req, res, next) => {
  try {
    const { topic, language, difficulty } = validate(quizSchema, req.body);
    log("INFO", "quiz_generating", { userId: req.user?.id ?? "anon", topic, difficulty });
    const raw  = await callBedrock(
      `You are an expert CS educator and exam paper setter.

Task: Generate exactly 10 quiz questions on the given topic: 4 MCQ, 3 ShortAnswer, 3 Coding.

Respond ONLY with valid JSON (no markdown, no fences).
Schema: {"topic":string,"language":string,"difficulty":string,"totalMarks":number,"questions":[{"id":number,"type":"MCQ"|"ShortAnswer"|"Coding","question":string,"difficulty":"Easy"|"Medium"|"Hard","marks":number,"options":string[]|null,"correctAnswer":string,"explanation":string,"hint":string}]}

Rules:
- MCQ: exactly 4 options A) B) C) D). correctAnswer is the letter. All distractors must be plausible — no obviously absurd options.
- ShortAnswer: expect 2-4 sentence answers. correctAnswer should be a model answer.
- Coding: include a clear problem statement with sample input/output in the question text. correctAnswer should be working code.
- Bloom's taxonomy spread: at least 2 questions testing recall/understanding, at least 3 testing application/analysis, and at least 2 testing synthesis/evaluation.
- Marks: MCQ=1, Short=3, Coding=5. totalMarks must equal the sum.
- Each question must be distinct — no repeated patterns, no trivial variations of the same question.
- Explanations should teach WHY the answer is correct, not just restate it.
- Hints should nudge toward the concept, not reveal the answer.`,
      `Topic: ${topic}\nLanguage: ${language}\nDifficulty: ${difficulty}`,
      4096
    );
    const quiz = parseAIJson(raw);
    if (req.user?.id) {
      saveProgress(req.user.id, `quiz_${topic}`, { attempted: true, topic, difficulty, ts: new Date().toISOString() })
        .catch(e => log("WARN", "quiz_progress_failed", { error: e.message }));
    }
    log("INFO", "quiz_generated", { topic, count: quiz.questions?.length });
    res.json({ success: true, data: { quiz } });
  } catch (e) { next(e); }
});

app.post("/api/viva", upload.single("file"), async (req, res, next) => {
  try {
    if (!req.file) throw appError(400, "Please upload a code file");
    const code = req.file.buffer.toString("utf-8");
    if (code.trim().length < 10) throw appError(400, "File appears empty");
    const lang = req.file.originalname.split(".").pop().toLowerCase();
    if (req.user?.id) {
      uploadToS3(req.file.buffer, req.file.originalname, req.user.id, "code")
        .catch(e => log("WARN", "viva_s3_failed", { error: e.message }));
    }
    log("INFO", "viva_generating", { userId: req.user?.id ?? "anon", lang });
    const raw = await callBedrock(
      `You are an expert CS professor conducting a viva voce examination.

Task: Analyze the submitted code thoroughly and generate exactly 20 viva questions. NEVER reveal answers in the questions or hints.

Respond ONLY with valid JSON (no markdown, no fences).
Schema: {"language":string,"summary":string,"overallDifficulty":string,"evaluationCriteria":[{"criterion":string,"weight":number,"description":string}],"questions":[{"id":number,"question":string,"difficulty":"Easy"|"Medium"|"Hard","category":"Concept"|"Implementation"|"Logic"|"Debugging"|"Optimization"|"Theory","hint":string,"marks":number,"followUp":string}]}

Rules:
- Mix: ~6 Easy, ~9 Medium, ~5 Hard.
- Questions MUST be specific to the actual submitted code — reference specific functions, variables, line logic, or design choices. Generic CS questions are not acceptable.
- Include at least 3 "Why did you choose this approach?" or "What alternatives did you consider?" style questions.
- Include at least 2 "What would happen if...?" hypothetical questions about the code.
- followUp: a natural follow-up question the examiner could ask based on the student's likely answer.
- Hints should guide the student's thinking without giving away the answer.
- Cover all categories: Concept, Implementation, Logic, Debugging, Optimization, Theory.
- Questions should progress from surface-level understanding to deep comprehension.`,
      `Filename: ${req.file.originalname}\nLanguage: ${lang}\n\nCode:\n\`\`\`${lang}\n${code.slice(0, 8000)}\n\`\`\``,
      4096
    );
    log("INFO", "viva_generated", { userId: req.user?.id ?? "anon", lang });
    res.json({ success: true, data: parseAIJson(raw) });
  } catch (e) { next(e); }
});

const debugSchema = Joi.object({
  code:     Joi.string().trim().min(2).max(10000).required(),
  language: Joi.string().trim().max(50).default("python"),
  error:    Joi.string().trim().max(2000).allow("").default(""),
});

app.post("/api/debug", async (req, res, next) => {
  try {
    const { code, language, error: errMsg } = validate(debugSchema, req.body);
    log("INFO", "debug_request", { userId: req.user?.id ?? "anon", language });
const raw = await callBedrock(
  `You are a warm, encouraging Socratic coding mentor on BharatLearn.

Task: Analyze the submitted code thoroughly and help the student learn from their mistakes.

Respond ONLY with valid JSON (no markdown, no fences).
Schema: {"language":string,"hasError":boolean,"errorType":string,"errorSummary":string,"conceptExplanation":string,"hints":["hint1","hint2","hint3"],"affectedLines":number[],"difficulty":"Easy"|"Medium"|"Hard","commonMistake":string,"furtherReading":string,"whatToSearchNext":string}

Rules:
- If the code is CORRECT with NO bugs:
  Set hasError=false, errorType="None", errorSummary="Code is correct! No bugs found."
  Provide 3 improvement hints: error handling, edge cases, performance, or readability.
  Set affectedLines=[].

- If bugs exist:
  1. errorSummary: one clear sentence describing the bug.
  2. conceptExplanation: explain the underlying concept the student likely misunderstands. Use an analogy if helpful.
  3. hints: exactly 3 progressive hints — (1) surface-level nudge, (2) conceptual clue, (3) near-solution guidance. NEVER include the corrected code in any hint.
  4. affectedLines: list the 1-based line numbers where the bug occurs.
  5. commonMistake: describe why this mistake is common so the student doesn't feel bad.
  6. whatToSearchNext: a Google/Stack Overflow search query the student can use to learn more (e.g., "python off-by-one error in for loop").
  7. furtherReading: a topic or keyword to study, not a URL.

- Handle partial code snippets or multi-function code gracefully — analyze what is provided without complaining about missing context.
- Be warm and educational throughout. Encourage the student.`,
  `Language: ${language}\nCode:\n\`\`\`${language}\n${code}\n\`\`\`${errMsg ? `\nError: ${errMsg}` : ""}`,
  2048
);
    res.json({ success: true, data: parseAIJson(raw) });
  } catch (e) { next(e); }
});

// ─── 404 + Error Handler ──────────────────────────────────────────────────────
app.use((req, _res, next) =>
  next(appError(404, `Route not found: ${req.method} ${req.originalUrl}`, "NOT_FOUND"))
);

// eslint-disable-next-line no-unused-vars
app.use((err, req, res, _next) => {
  const status = err.status ?? 500;
  log(status >= 500 ? "ERROR" : "WARN", "request_error", {
    method: req.method, url: req.url, status, message: err.message, requestId: req.id,
  });
  res.status(status).json({
    success: false,
    error: { code: err.code ?? "SERVER_ERROR", message: (err.isOp || !IS_PROD) ? err.message : "An unexpected error occurred" },
    meta:  { requestId: req.id, ts: new Date().toISOString() },
  });
});

// ─── Start ────────────────────────────────────────────────────────────────────
const server = app.listen(PORT, async () => {
  if (process.env.AWS_ACCESS_KEY_ID)
    _ensureLogStream().catch(e => console.warn("[CW] Log stream init failed:", e.message));

  const ok = "✅", warn = "⚠️ ";
  console.log(`\n🚀  BharatLearn API  →  http://localhost:${PORT}`);
  console.log(`    Google OAuth   : ${GOOGLE_CLIENT_ID           ? ok : warn} ${GOOGLE_CLIENT_ID           ? "configured"    : "not configured"}`);
  console.log(`    Groq AI  : ${process.env.GROQ_API_KEY ? ok : warn} ${process.env.GROQ_API_KEY ? "configured" : "not configured"}`);
  console.log(`    Model    : ${ok} ${GROQ_MODEL}`);
  console.log(`    DynamoDB       : ${process.env.AWS_ACCESS_KEY_ID ? ok : warn} ${process.env.AWS_ACCESS_KEY_ID ? "ready"       : "needs AWS keys"}`);
  console.log(`    S3 Bucket      : ${ok} ${S3_BUCKET}`);
  console.log(`    CloudWatch     : ${process.env.AWS_ACCESS_KEY_ID ? ok : warn} ${process.env.AWS_ACCESS_KEY_ID ? CW_LOG_GROUP  : "needs AWS keys"}\n`);
  log("INFO", "server_started", { port: PORT, env: process.env.NODE_ENV });
});

["SIGTERM","SIGINT"].forEach(sig =>
  process.on(sig, () => {
    log("INFO", "server_shutdown", { signal: sig });
    clearInterval(_cwFlushTimer);
    _flushCWLogs().finally(() => server.close(() => { console.log("Server closed."); process.exit(0); }));
  })
);
process.on("unhandledRejection", r => { log("ERROR", "unhandled_rejection", { reason: String(r) }); process.exit(1); });

export default server;
