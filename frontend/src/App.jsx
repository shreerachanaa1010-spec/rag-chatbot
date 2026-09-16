import { useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';

const regions = [
  { value: '', label: 'All regions' },
  { value: 'US', label: 'United States' },
  { value: 'EU', label: 'European Union' },
  { value: 'Global', label: 'Global' },
];

function StatusPill({ ready }) {
  return (
    <div className={`status ${ready ? 'ready' : ''}`}>
      <span /> <b>{ready ? 'Policy Library Ready' : 'Policy Library Unavailable'}</b>
    </div>
  );
}

function SourceList({ sources }) {
  return (
    <details className="sources">
      <summary>Retrieved source chunks</summary>
      {sources.map((item) => (
        <article className="source" key={item.chunk.chunk_id}>
          <strong>{item.chunk.doc_id} {item.chunk.version}</strong>
          <small>{item.chunk.region} · distance {item.distance.toFixed(3)}</small>
          <p>{item.chunk.text}</p>
        </article>
      ))}
    </details>
  );
}

function Result({ data }) {
  const action = data.next_action === 'send_to_employee' ? '🚀 Send to employee' : '🚀 HR review';
  const [emailStatus, setEmailStatus] = useState('');
  const confidence = data.confidence === 'high' ? 'High Confidence' : 'Needs Review';

  async function sendToHr() {
    setEmailStatus('Sending...');
    try {
      const response = await fetch('/api/send-to-hr', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: data.question,
          answer: data.answer,
          draft_email: data.draft_email,
          cited_sections: data.cited_sections,
        }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || 'Unable to send the case to HR.');
      setEmailStatus(result.message);
    } catch (requestError) {
      setEmailStatus(requestError.message);
    }
  }

  return (
    <div className="result">
      <div className="result-heading">
        <div>
          <p className="eyebrow">STRUCTURED DECISION</p>
          <p className="result-kicker">Policy answer</p>
        </div>
        <span className={`confidence-badge ${data.confidence === 'high' ? 'high' : 'review'}`}>{confidence}</span>
        <span className={`badge ${data.conflict_flag || data.confidence !== 'high' ? 'alert' : ''}`}>
          {data.conflict_flag ? 'Version conflict · review required' : 'Grounded answer'}
        </span>
      </div>
      {data.older_version_warning && <div className="version-warning">{data.older_version_warning}</div>}
      <div className="answer-text">
        <ReactMarkdown>{data.answer}</ReactMarkdown>
      </div>
      <div className="meta-grid">
        <div className="meta-item"><small>Confidence</small><strong className={data.confidence === 'high' ? 'confidence-high' : 'confidence-review'}>{confidence}</strong></div>
        <div className="meta-item"><small>Next action</small><strong>{action}</strong></div>
        <div className="meta-item"><small>Evidence</small><strong>{data.retrieved.length} sources</strong></div>
      </div>
      <div className="section-block">
        <h3>Citations</h3>
        <ul>
          {data.cited_sections.length ? data.cited_sections.map((citation) => <li key={citation}>{citation}</li>) : <li>No citations returned.</li>}
        </ul>
      </div>
      <div className="section-block email-block">
        <h3>Draft response</h3>
        <p className="draft-email">{data.draft_email}</p>
        {data.next_action === 'escalate_to_hr_review' && (
          <div className="hr-action">
            <button type="button" onClick={sendToHr} disabled={emailStatus === 'Sending...'}>
              <span>{emailStatus === 'Sending...' ? 'Sending to HR...' : 'Send case to HR'}</span><strong>↗</strong>
            </button>
            {emailStatus && <p className="action-status">{emailStatus}</p>}
          </div>
        )}
      </div>
      <SourceList sources={data.retrieved} />
    </div>
  );
}

export default function App() {
  const [question, setQuestion] = useState('');
  const [recentQuestions, setRecentQuestions] = useState([]);
  const [region, setRegion] = useState('');
  const [health, setHealth] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch('/api/health')
      .then((response) => response.json())
      .then(setHealth)
      .catch(() => setHealth({ index_ready: false, offline: true }));
  }, []);

  async function submitQuestion(event) {
    event.preventDefault();
    const trimmedQuestion = question.trim();
    const wordCount = trimmedQuestion.match(/[A-Za-z][A-Za-z'-]*/g)?.length ?? 0;
    if (wordCount < 2 || (wordCount < 3 && !trimmedQuestion.includes('?'))) {
      setResult(null);
      setError('Please enter a specific HR policy question, including the relevant policy area or location.');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);
    try {
      const response = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: trimmedQuestion, region: region || null }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'The policy desk could not answer this question.');
      setResult(data);
      setRecentQuestions((previous) => [trimmedQuestion, ...previous.filter((item) => item !== trimmedQuestion)].slice(0, 4));
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  const backendOffline = health?.offline;
  return (
    <main className="shell">
      <nav className="nav-bar" aria-label="Primary navigation">
        <div className="brand"><span className="brand-mark">P</span><span>POLICY DESK</span></div>
        <span className="nav-caption">HR OPERATIONS <i /> RESOLUTION ENGINE</span>
      </nav>
      <header className="topbar">
        <div>
          <p className="eyebrow">🔍 SEARCH / 01</p>
          <h1>Clear answers for <em>complex policies.</em></h1>
          <p className="intro">Search the policy library, review the evidence, and route uncertainty to HR.</p>
        </div>
        <StatusPill ready={!backendOffline && health?.index_ready} />
      </header>

      <section className="workspace">
        <form className="question-panel" onSubmit={submitQuestion}>
          <div className="panel-heading"><span className="step-number">🔍</span><div><p className="eyebrow">SEARCH / 01</p><h2>What do you need to know?</h2></div></div>
          <label htmlFor="question">Employee question</label>
          <textarea id="question" minLength="3" maxLength="2000" required value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Try: How much parental leave am I eligible for in the US?&#10;Or: When can I apply for PTO?" />
          <div className="textarea-meta"><span>Be specific about location or policy</span><span>{question.length}/2000</span></div>
          <div className="form-row">
            <label className="region-field" htmlFor="region">Region
              <select id="region" value={region} onChange={(event) => setRegion(event.target.value)}>
                {regions.map((item) => <option value={item.value} key={item.value}>{item.label}</option>)}
              </select>
            </label>
            <button type="submit" disabled={loading}><span>{loading ? 'Searching...' : 'Ask policy desk'}</span><strong>↗</strong></button>
          </div>
          <p className="form-note"><span className="note-dot" /> 📑 Review follows every search. Conflicting versions are escalated automatically.</p>
          {recentQuestions.length > 0 && (
            <div className="recent-questions">
              <div className="recent-heading"><span>Recent Questions</span><small>THIS SESSION</small></div>
              {recentQuestions.map((item) => (
                <button type="button" className="recent-question" key={item} onClick={() => setQuestion(item)}>
                  <span>{item}</span><strong>↗</strong>
                </button>
              ))}
            </div>
          )}
        </form>

        <section className="answer-panel" aria-live="polite">
          {!result && !loading && !error && <div className="empty-state"><div className="signal">+</div><p className="eyebrow">YOUR ANSWER WILL APPEAR HERE</p><h2>Start with a question.</h2><p>Every response is paired with citations, confidence, and the source excerpts behind it.</p></div>}
          {loading && <div className="loading"><span /><span /><span /><p>Searching policies and checking versions...</p></div>}
          {error && <div className="error">{error}</div>}
          {result && <Result data={result} />}
        </section>
      </section>
      <footer>🔍 Search <span>·</span> 📑 Review <span>·</span> 🚀 Escalate</footer>
    </main>
  );
}
