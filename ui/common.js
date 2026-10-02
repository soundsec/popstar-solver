/* 三个页面共用的错误呈现与连接检查。

为什么需要它：异步处理函数里一个未捕获的 rejection 在浏览器里是**完全静默**的
——界面上没有任何变化，用户只会觉得「点击没反应」。本模块把失败显式化：

* 启动时先探一次服务，连不上就在页顶显示红条；
* 所有异步处理函数用 ``UI.guard`` 包一层，出错写到状态栏 + 页顶红条；
* 全局 error / unhandledrejection 兜底，任何漏网的异常都能被看见。
*/

(function () {
  'use strict';

  const $ = (id) => document.getElementById(id);

  function banner(message, kind) {
    let el = $('fatal');
    if (!el) {
      el = document.createElement('div');
      el.id = 'fatal';
      document.body.insertBefore(el, document.body.firstChild);
    }
    el.className = 'fatal' + (kind ? ' ' + kind : '');
    el.textContent = message;
    el.hidden = false;
  }

  function clearBanner() {
    const el = $('fatal');
    if (el) el.hidden = true;
  }

  /** 探测本地服务是否在跑；连不上时给出可操作的提示。 */
  async function ping() {
    try {
      const resp = await fetch('/api/generators', { cache: 'no-store' });
      if (!resp.ok) throw new Error('HTTP ' + resp.status);
      clearBanner();
      return true;
    } catch (err) {
      banner(
        '无法连接本地服务（' + err.message + '）。' +
        '请先运行「启动.bat」或 python start.py，然后刷新本页。',
        'bad'
      );
      return false;
    }
  }

  /** 包装异步事件处理函数：出错时同时写状态栏与页顶红条。 */
  function guard(handler, statusId) {
    return async function wrapped(...args) {
      try {
        return await handler.apply(this, args);
      } catch (err) {
        const message = (err && err.message) ? err.message : String(err);
        if (statusId && $(statusId)) $(statusId).textContent = '出错：' + message;
        banner('操作失败：' + message, 'bad');
      }
    };
  }

  async function api(path, body) {
    let resp;
    try {
      resp = await fetch(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body || {}),
      });
    } catch (err) {
      throw new Error('无法连接本地服务（' + err.message + '）');
    }
    let data;
    try {
      data = await resp.json();
    } catch (err) {
      throw new Error('服务返回了非 JSON 响应（HTTP ' + resp.status + '）');
    }
    if (data.error) throw new Error(data.error);
    return data;
  }

  window.addEventListener('error', (e) => {
    banner('脚本错误：' + e.message, 'bad');
  });
  window.addEventListener('unhandledrejection', (e) => {
    const reason = e.reason;
    banner('未处理的异步错误：' + ((reason && reason.message) || reason), 'bad');
  });

  window.UI = { $, banner, clearBanner, ping, guard, api };
})();
