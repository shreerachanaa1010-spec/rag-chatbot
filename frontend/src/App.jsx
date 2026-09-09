import { useEffect, useState } from 'react';

const regions = [
  { value: '', label: 'All regions' },
  { value: 'US', label: 'United States' },
  { value: 'EU', label: 'European Union' },
  { value: 'Global', label: 'Global' },
];

function StatusPill({ ready }) {
  return (
    <div className={`status ${ready ? 'ready' : ''}`}>
      <span /> {ready ? 'Index ready' : 'Index not built'}
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
  const action = data.next_action === 'send_to_employee' ? 'Send to employee' : 'HR review';
  return (
    <div className="result">
      <div className="result-heading">
        <p className="eyebrow">STRUCTURED DECISION</p>
        <span className={`badge ${data.conflict_flag ? 'alert' : ''}`}>
          {data.conflict_flag ? 'Version conflict · review required' : 'Grounded answer'}
        </span>
      </div>
      {data.older_version_warning && <div className="version-warning">{data.older_version_warning}</div>}
      <h2>{data.answer}</h2>
      <div className="meta-grid">
        <div><small>Confidence</small><strong>{data.confidence}</strong></div>
        <div><small>Next action</small><strong>{action}</strong></div>
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
      </div>
      <SourceList sources={data.retrieved} />
    </div>
  );
}

export default function App() {
  const [question, setQuestion] = useState('');
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
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const response = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: question.trim(), region: region || null }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'The policy desk could not answer this question.');
      setResult(data);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  const backendOffline = health?.offline;
  return (
    <main className="shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">HR OPERATIONS / POLICY DESK</p>
          <h1>Find the policy answer.<br /><em>Keep the evidence.</em></h1>
        </div>
        <StatusPill ready={!backendOffline && health?.index_ready} />
      </header>

      <section className="workspace">
        <form className="question-panel" onSubmit={submitQuestion}>
          <label htmlFor="question">Employee question</label>
          <textarea id="question" minLength="3" maxLength="2000" required value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="e.g. How much parental leave am I eligible for in the US?" />
          <div className="form-row">
            <label className="region-field" htmlFor="region">Region
              <select id="region" value={region} onChange={(event) => setRegion(event.target.value)}>
                {regions.map((item) => <option value={item.value} key={item.value}>{item.label}</option>)}
              </select>
            </label>
            <button type="submit" disabled={loading}><span>{loading ? 'Searching...' : 'Ask policy desk'}</span><strong>↗</strong></button>
          </div>
          <p className="form-note">Answers use indexed policy text only. Conflicting versions are escalated automatically.</p>
        </form>

        <section className="answer-panel" aria-live="polite">
          {!result && !loading && !error && <div className="empty-state"><div className="signal">◎</div><h2>Ready when you are.</h2><p>Ask a policy question to see a cited decision and the source excerpts behind it.</p></div>}
          {loading && <div className="loading"><span /><span /><span /><p>Searching policies and checking versions...</p></div>}
          {error && <div className="error">{error}</div>}
          {result && <Result data={result} />}
        </section>
      </section>
      <footer>React frontend <span>·</span> FastAPI backend <span>·</span> grounded policy retrieval</footer>
    </main>
  );
}
