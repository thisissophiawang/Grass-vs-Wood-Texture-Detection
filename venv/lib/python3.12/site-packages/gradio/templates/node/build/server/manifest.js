const manifest = (() => {
function __memo(fn) {
	let value;
	return () => value ??= (value = fn());
}

return {
	appDir: "_app",
	appPath: "_app",
	assets: new Set([]),
	mimeTypes: {},
	_: {
		client: {"start":"_app/immutable/entry/start.CO3xjxZc.js","app":"_app/immutable/entry/app.4hQdf3To.js","imports":["_app/immutable/entry/start.CO3xjxZc.js","_app/immutable/chunks/client.PjzZEX3L.js","_app/immutable/entry/app.4hQdf3To.js","_app/immutable/chunks/preload-helper.DpQnamwV.js"],"stylesheets":[],"fonts":[],"uses_env_dynamic_public":false},
		nodes: [
			__memo(() => import('./chunks/0-DTUnaKEU.js')),
			__memo(() => import('./chunks/1-CztSoRSS.js')),
			__memo(() => import('./chunks/2-CI5fLukF.js').then(function (n) { return n.ax; }))
		],
		routes: [
			{
				id: "/[...catchall]",
				pattern: /^(?:\/(.*))?\/?$/,
				params: [{"name":"catchall","optional":false,"rest":true,"chained":true}],
				page: { layouts: [0,], errors: [1,], leaf: 2 },
				endpoint: null
			}
		],
		matchers: async () => {
			
			return {  };
		},
		server_assets: {}
	}
}
})();

const prerendered = new Set([]);

const base = "";

export { base, manifest, prerendered };
//# sourceMappingURL=manifest.js.map
