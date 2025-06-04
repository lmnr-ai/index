([x, y]) => {
    const elementWeights = {
        'button': 10,
        'a': 10,
        'input': 10,
        'select': 10,
        'textarea': 10,
        'summary': 8,
        'details': 7,
        'label': 5, // Labels are clickable but not always interactive
        'option': 7,
        'tr': 4,
        'th': 3,
        'td': 3,
        'li': 8,
        'div': 2,
        'span': 1,
        'img': 2,
        'svg': 3,
        'path': 3
    };

    function getAdjustedBoundingClientRect(element, contextInfo = null) {
        const rect = element.getBoundingClientRect();
        
        // If element is in an iframe, adjust coordinates
        if (contextInfo && contextInfo.iframe) {
            const iframeRect = contextInfo.iframe.getBoundingClientRect();
            return {
                top: rect.top + iframeRect.top,
                right: rect.right + iframeRect.left,
                bottom: rect.bottom + iframeRect.top,
                left: rect.left + iframeRect.left,
                width: rect.width,
                height: rect.height
            };
        }
        
        return rect;
    }

    function generateUniqueId() {
        const rand = Math.random().toString(36);
        return `cv-${rand}`;
    }

    function getEffectiveZIndex(element) {
        let current = element;
        let zIndex = 'auto';
        
        while (current && current !== document) {
            const style = window.getComputedStyle(current);
            if (style.position !== 'static' && style.zIndex !== 'auto') {
                zIndex = parseInt(style.zIndex, 10);
                break;
            }
            current = current.parentElement;
        }
        
        return zIndex === 'auto' ? 0 : zIndex;
    }

    function getElement(element, contextInfo = null) {
        const rect = getAdjustedBoundingClientRect(element, contextInfo);

        // Get element text (direct or from children)
        let text = element.innerText || '';
        if (!text) {
            const textNodes = Array.from(element.childNodes)
                .filter(node => node.nodeType === Node.TEXT_NODE)
                .map(node => node.textContent.trim())
                .filter(content => content.length > 0);
            text = textNodes.join(' ');
        }

        // Ensure each element has a index_id
        let browserId = element.getAttribute('data-browser-agent-id');

        if (!browserId) {
            const uniqueId = generateUniqueId();
            element.setAttribute('data-browser-agent-id', uniqueId);
            browserId = uniqueId;
        }

        // Extract important attributes
        const attributes = {};
        ['id', 'class', 'href', 'type', 'name', 'value', 'placeholder', 'aria-label', 'title', 'role', 'readonly'].forEach(attr => {
            if (element.hasAttribute(attr)) {
                attributes[attr] = element.getAttribute(attr);
            }
        });

        let elementType = element.tagName.toLowerCase();
        let inputType = null;

        // Handle input elements specifically
        if (elementType === 'input' && element.hasAttribute('type')) {
            inputType = element.getAttribute('type').toLowerCase();
        }

        const index = 0; // Replaced by cv index in python code
        let weight = elementWeights[elementType] || 1;

        const elementData = {
            tagName: elementType,
            text: text.trim(),
            attributes,
            index,
            weight: weight,
            browserAgentId: browserId,  // Use the guaranteed ID
            inputType: inputType,  // Add specific input type
            viewport: {
                x: Math.round(rect.left),
                y: Math.round(rect.top),
                width: Math.round(rect.width),
                height: Math.round(rect.height)
            },
            page: {
                x: Math.round(rect.left + window.scrollX),
                y: Math.round(rect.top + window.scrollY),
                width: Math.round(rect.width),
                height: Math.round(rect.height)
            },
            center: {
                x: Math.round(rect.left + rect.width/2),
                y: Math.round(rect.top + rect.height/2)
            },
            rect: {
                left: Math.round(rect.left),
                top: Math.round(rect.top),
                right: Math.round(rect.right),
                bottom: Math.round(rect.bottom),
                width: Math.round(rect.width),
                height: Math.round(rect.height)
            },
            zIndex: getEffectiveZIndex(element)
        };

        return elementData;
    };

    // Gets the element at the specified coordinates
    let element = document.elementFromPoint(x, y);

    // If no element is found, return false directly
    if (!element) {
        return false;
    }

    // Check whether elements can interact
    // - input, button, select, textarea, a, label, ...
    const interactiveTags = ['input', 'button', 'select', 'textarea', 'a', 'label', 'option', 'span'];
    if (interactiveTags.includes(element.tagName.toLowerCase())) {
        // return true;
        return getElement(element);
    }

    if (element.tagName.toLowerCase() == 'iframe') {
        const frameDocument = element.contentDocument;
        const rect = element.getBoundingClientRect();

        const element2 = frameDocument.elementFromPoint(x - rect.left, y - rect.top);
        
        if (interactiveTags.includes(element2.tagName.toLowerCase())) {
            return getElement(element2, {iframe: element, shadowHost: null});
        }
    }

    return null;
}