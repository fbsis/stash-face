// Local-only mock derived from Stash v0.31.1 GraphQL source, NOT a cloud service.
// No real photos, identities, credentials or facial models. Blank 128x128 PNG.
const http = require("node:http");
const image = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAIAAAACACAIAAABMXPacAAABKklEQVR4nO3RMQ0AAAjAMPybBhk9WBUs2Wyo0QHfNQBrANYArAFYA7AGYA3AGoA1AGsA1gCsAVgDsAZgDcAagDUAawDWAKwBWAOwBmANwBqANQBrANYArAFYA7AGYA3AGoA1AGsA1gCsAVgDsAZgDcAagDUAawDWAKwBWAOwBmANwBqANQBrANYArAFYA7AGYA3AGoA1AGsA1gCsAVgDsAZgDcAagDUAawDWAKwBWAOwBmANwBqANQBrANYArAFYA7AGYA3AGoA1AGsA1gCsAVgDsAZgDcAagDUAawDWAKwBWAOwBmANwBqANQBrANYArAFYA7AGYA3AGoA1AGsA1gCsAVgDsAZgDcAagDUAawDWAKwBWAOwBmANwBqANQBrANYArAFYA7AGYAeU6UsysMvN4wAAAABJRU5ErkJggg==", "base64");
const KEY = "stashReview:state";
const fresh = () => ({revision: 1, metadata: 1, mode: "ok", performers: ["7"], mutations: 0, requests: 0});
function send(res, code, body) {
  res.writeHead(code, {"Content-Type": "application/json"});
  res.end(JSON.stringify(body));
}
function body(req) {
  if (req.body !== undefined) return Promise.resolve(req.body);
  return new Promise(resolve => { let raw = ""; req.on("data", c => raw += c);
    req.on("end", () => { try { resolve(JSON.parse(raw || "{}")); } catch { resolve({}); } }); });
}
http.createServer(async (req, res) => {
  const input = body(req); // attach before awaiting pm.state
  try {
    const path = new URL(req.url, "http://mock").pathname;
    let state = await pm.state.get(KEY) || fresh();
    if (path === "/test/reset" && req.method === "POST") {
      await pm.state.set(KEY, fresh()); return send(res, 200, fresh());
    }
    if (path === "/test/config" && req.method === "POST") {
      const changes = await input;
      for (const key of ["revision", "metadata", "mode", "performers", "deleted"]) if (key in changes) state[key] = changes[key];
      await pm.state.set(KEY, state); return send(res, 200, state);
    }
    if (path === "/test/state") return send(res, 200, state);
    if (req.headers.apikey !== "mock-api-key") return send(res, 401, {error: "Invalid Stash API key"});
    state.requests++; await pm.state.set(KEY, state);
    if (state.mode === "unavailable") return send(res, 503, {error: "Temporarily unavailable"});
    if (state.mode === "html-error") { res.writeHead(500, {"Content-Type":"text/html"}); return res.end("<h1>upstream down</h1>"); }
    if (state.mode === "timeout") await new Promise(resolve => setTimeout(resolve, 1500));
    if (path === "/image/1") {
      if (state.mode === "redirect-image") { res.writeHead(302, {Location: "http://127.0.0.1:1/private"}); return res.end(); }
      if (state.mode === "oversized-image") {
        res.writeHead(200, {"Content-Type":"image/png", "Content-Length": String(image.length)}); return res.end(image);
      }
      res.writeHead(200, {"Content-Type":"image/png"}); return res.end(image);
    }
    if (path !== "/graphql" || req.method !== "POST") return send(res, 501, {error: "Endpoint not modeled"});
    const {query = "", variables: v = {}} = await input;
    if (state.mode === "graphql-error") return send(res, 200, {errors: [{message: "Example GraphQL failure"}]});
    if (state.mode === "invalid-json") { res.writeHead(200); return res.end("not json"); }
    const operation = /\b(?:query|mutation)\s+(\w+)/.exec(query)?.[1];
    const photo = {id:"1",title:"Synthetic review fixture",updated_at:"metadata-" + state.metadata,
      paths:{image:state.mode === "cross-origin" ? "http://127.0.0.1:1/private" : "/stash/image/1"},
      performers:state.performers.map(id=>({id})),
      visual_files:[{__typename:"ImageFile",id:"11",mod_time:"revision-" + state.revision,size:100,
        fingerprints:[{type:"md5",value:"synthetic-" + state.revision}]}]};
    let data;
    if (operation === "ReviewConnection") data = {version:{version:"v0.31.1-mock"}};
    else if (operation === "ReviewImages") {
      if (!Number.isInteger(v.page) || v.page < 1) return send(res, 400, {error:"Invalid page"});
      data = {findImages:{count:state.deleted?0:1,images:v.page===1&&!state.deleted?[photo]:[]}};
    } else if (operation === "ReviewImage") data = {findImage:v.id==="1"&&!state.deleted?photo:null};
    else if (operation === "ReviewPerformers") data = {findPerformers:{count:1,performers:[{id:"9",name:"Manually selected Performer"}]}};
    else if (operation === "ReviewPerformer") data = {findPerformer:["7","9"].includes(v.id)?{id:v.id,name:"Manual Performer"}:null};
    else if (operation === "ReviewAddPerformers") {
      const update = v.input;
      if (!update || JSON.stringify(update.ids)!=='["1"]' || update.performer_ids?.mode!=="ADD"
          || !Array.isArray(update.performer_ids.ids) || Object.keys(update).some(k=>!["ids","performer_ids"].includes(k)))
        return send(res, 400, {error:"Only additive, explicit Performer updates are allowed"});
      state.performers = [...new Set([...state.performers,...update.performer_ids.ids])];
      state.mutations++; state.metadata++;
      await pm.state.set(KEY, state);
      data = {bulkImageUpdate:[{id:"1",performers:state.performers.map(id=>({id}))}]};
    } else return send(res, 501, {error:"GraphQL operation not modeled"});
    return send(res, 200, {data});
  } catch (error) { return send(res, 500, {error: "Mock handler failed"}); }
}).listen(process.env.PORT || 49171);
