const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const {JSDOM} = require('jsdom');
const root = require('node:path').resolve(__dirname, '..');
const token = 'synthetic-test-token-' + 'x'.repeat(32);
const settle = () => new Promise(resolve => setImmediate(resolve));

function fixture() {
  const dom = new JSDOM(fs.readFileSync(root + '/stash_face/static/index.html', 'utf8'),
    {url:'http://localhost:8765/',runScripts:'outside-only'});
  const {window} = dom;
  const requests = [], revoked = [];
  window.URL.createObjectURL = () => 'blob:synthetic';
  window.URL.revokeObjectURL = url => revoked.push(url);
  window.confirm = () => false;
  window.fetch = async (path, init) => {
    requests.push({path:String(path), ...init});
    let data;
    if(path==='/health') data = {stash_url:'http://stash.local:9999',worker_alive:true,detector:{provider:'test',actual_device:'CPU'}};
    else if(path==='/groups') data = [{id:1,name:'<script>unsafe()</script>',performer_id:'9',face_count:1,image_count:1}];
    else if(String(path).startsWith('/faces?')) data = {items:[{id:1,image_id:'1',title:'<img onerror=unsafe()>',group_id:1,group_name:'Manual'}],next_cursor:null};
    else if(String(path).endsWith('/crop')) return {ok:true,blob:async()=>new window.Blob(['test'])};
    else if(path==='/jobs') data = [];
    else if(path==='/assignments') data = {job:{id:'confirmed-job'}};
    else if(path==='/faces/review') data = {reviewed:1};
    else throw new Error('Unexpected fetch: ' + path);
    return {ok:true,json:async()=>data};
  };
  window.eval(fs.readFileSync(root + '/stash_face/static/app.js', 'utf8'));
  return {window,dom,requests,revoked,$:id=>window.document.getElementById(id)};
}
async function connect(f) {
  f.$('token').value = token;
  f.$('connect-form').dispatchEvent(new f.window.Event('submit',{bubbles:true,cancelable:true}));
  await settle(); await settle();
}

test('private crops use header auth, untrusted labels stay text, and tokens are not persisted',async()=>{
  const f = fixture();
  try {
    await connect(f);
    assert.equal(f.$('workspace').hidden,false);
    assert.equal(f.$('token').value,'');
    assert.equal(f.window.localStorage.length,0);
    assert.equal(f.window.document.querySelectorAll('[onerror]').length,0);
    assert.ok(f.$('faces').textContent.includes('<img onerror=unsafe()>'));
    const crop = f.requests.find(r=>r.path.endsWith('/crop'));
    assert.equal(crop.headers['X-API-Token'],token);
    assert.ok(!crop.path.includes(token));
    assert.equal(f.$('faces').querySelector('img').src,'blob:synthetic');
    assert.equal(f.requests.filter(r=>r.path==='/assignments').length,0);
    f.$('disconnect').click();
    assert.equal(f.$('workspace').hidden,true);
    assert.equal(f.$('faces').textContent,'');
    assert.ok(f.revoked.includes('blob:synthetic'));
  } finally {f.dom.window.close();}
});

test('Performer application requires selected faces and affirmative human confirmation',async()=>{
  const f = fixture();
  try {
    await connect(f);
    f.$('apply').click(); await settle();
    assert.equal(f.requests.filter(r=>r.path==='/assignments').length,0);
    f.$('select-all').click();
    f.$('apply').click(); await settle();
    assert.equal(f.requests.filter(r=>r.path==='/assignments').length,0);
    f.window.confirm = () => true;
    f.$('apply').click(); await settle(); await settle();
    const writes = f.requests.filter(r=>r.path==='/assignments');
    assert.equal(writes.length,1);
    assert.deepEqual(JSON.parse(writes[0].body),{face_ids:[1],confirmed:true});
  } finally {f.dom.window.close();}
});

test('manual grouping is local review and does not call the assignment endpoint',async()=>{
  const f = fixture();
  try {
    await connect(f);
    f.$('select-all').click();
    f.$('target-group').value='1';
    f.$('move').click(); await settle(); await settle();
    assert.equal(f.requests.filter(r=>r.path==='/assignments').length,0);
    const review = f.requests.find(r=>r.path==='/faces/review');
    assert.deepEqual(JSON.parse(review.body),{face_ids:[1],group_id:1,ignored:false});
  } finally {f.dom.window.close();}
});

test('a late crop response cannot repopulate the dashboard after disconnect',async()=>{
  const f = fixture();
  try {
    const normalFetch = f.window.fetch;
    let release;
    f.window.fetch = (path, options) => String(path).endsWith('/crop')
      ? new Promise(resolve=>{release=resolve;}) : normalFetch(path,options);
    await connect(f);
    assert.equal(typeof release,'function');
    f.$('disconnect').click();
    release({ok:true,blob:async()=>new f.window.Blob(['test'])});
    await settle(); await settle();
    assert.equal(f.$('faces').textContent,'');
    assert.equal(f.$('workspace').hidden,true);
  } finally {f.dom.window.close();}
});
