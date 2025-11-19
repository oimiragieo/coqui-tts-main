# Coqui TTS Modernization Roadmap

## Quick Stats
- **Codebase Age**: 2+ years old
- **Size**: 2.9MB, 293 Python files, ~30,000 LOC
- **Models Supported**: 15+ TTS models, 8+ vocoders
- **Production Ready**: Yes, actively maintained
- **Current Python Support**: 3.9, 3.10, 3.11

---

## Critical Modernization Opportunities (Priority Order)

### 🔴 HIGH PRIORITY

#### 1. Unify Configuration System (Score: 10/10)
**Problem**: 
- Inconsistent config patterns: some use `config.model_args`, others use flat config
- Legacy `get_from_config_or_model_args()` workarounds scattered throughout
- Reduces type safety and IDE autocomplete
- Makes adding new models harder

**Solution**:
- Implement single config pattern for all models
- Migrate all to Python dataclasses (already using Coqpit)
- Add comprehensive type hints
- Create migration guide for existing model configs

**Impact**: 
- Easier model addition
- Better IDE support
- Reduced technical debt
- Estimated effort: 2-3 weeks

**Files Affected**: 
- TTS/config/__init__.py
- TTS/tts/configs/*.py (17 files)
- TTS/config/shared_configs.py

---

#### 2. Modernize to PyTorch 2.0+ (Score: 9/10)
**Problem**:
- Currently supports torch>=2.1 but doesn't leverage new features
- No use of torch.compile() for performance
- Missing torch.nn.functional patterns
- No structured tensor support

**Solution**:
- Adopt torch.compile() for model optimization
- Use torch.fx for symbolic execution
- Implement torch.nn.functional patterns
- Add structured tensor support for spectrogram handling
- Update attention implementations to use scaled_dot_product_attention

**Impact**:
- 20-40% inference speedup
- Better memory efficiency
- Future compatibility
- Estimated effort: 3-4 weeks

**Key Models**: VITS, XTTS, Tacotron2

---

#### 3. Add Comprehensive Type Hints (Score: 9/10)
**Problem**:
- Current codebase has sparse type hints
- No mypy/pyright validation
- IDE autocomplete often doesn't work
- Harder to maintain and debug

**Solution**:
- Add type hints to all public APIs
- Use mypy --strict mode
- Create type stubs for external dependencies
- Document type requirements in docstrings

**Impact**:
- Better developer experience
- Fewer runtime bugs
- Better IDE support
- Estimated effort: 2-3 weeks

**Starting Points**:
- TTS/api.py (public API)
- TTS/utils/synthesizer.py
- TTS/tts/models/base_tts.py

---

#### 4. Consolidate Duplicate Layer Implementations (Score: 8/10)
**Problem**:
- 12 different layer directories under TTS/tts/layers/
- Multiple implementations of same concepts (attention, normalization, etc.)
- Increased maintenance burden
- Code duplication

**Solution**:
- Create shared layer library
- Implement layer registry pattern
- Consolidate: Attention (8 variants), Normalization (5 variants), etc.
- Use composition instead of inheritance

**Impact**:
- Easier to maintain and debug
- Easier to add new models
- Smaller codebase
- Estimated effort: 3-4 weeks

**Affected Directories**:
- TTS/tts/layers/tacotron/
- TTS/tts/layers/vits/
- TTS/tts/layers/glow_tts/
- TTS/tts/layers/xtts/
- Plus: overflow, delightful_tts, etc.

---

### 🟡 MEDIUM PRIORITY

#### 5. Modernize Testing Infrastructure (Score: 7/10)
**Problem**:
- Using nose2 (old, less maintained)
- No property-based testing
- Test coverage unclear
- CI tests split across 14 workflows (hard to manage)

**Solution**:
- Migrate from nose2 → pytest (faster, better plugins)
- Add pytest-xdist for parallel testing
- Add hypothesis for property-based testing
- Consolidate CI workflows
- Target >80% code coverage

**Impact**:
- Faster test runs
- Better debugging
- More maintainable tests
- Estimated effort: 2-3 weeks

**Test Files**: 
- tests/ (14 categories, 100+ test files)

---

#### 6. Add Async/Streaming Inference API (Score: 7/10)
**Problem**:
- Current API is synchronous only
- No streaming support (needed for real-time apps)
- No batch inference optimization
- Memory inefficient for large batches

**Solution**:
- Create async inference interface
- Add streaming generator support
- Implement batch inference optimization
- Use asyncio + aiohttp for async server

**Impact**:
- Real-time TTS applications enabled
- Better performance for batch processing
- Server can handle more concurrent requests
- Estimated effort: 2-3 weeks

**Target API**:
```python
async def tts_async(text: str) → np.ndarray:
    pass

def tts_stream(text: str) → Iterator[np.ndarray]:
    pass
```

---

#### 7. Decouple from `trainer` Library (Score: 6/10)
**Problem**:
- Tight coupling to external `trainer>=0.0.32` library
- Hard to use models outside trainer framework
- Difficult to implement custom training loops
- Limits flexibility for research

**Solution**:
- Extract trainer-independent core
- Create lightweight trainer interface
- Support PyTorch Lightning, Hugging Face Trainer, custom loops
- Keep backward compatibility

**Impact**:
- More flexible for research
- Easier to integrate with other frameworks
- Estimated effort: 3-4 weeks

---

#### 8. Comprehensive API Documentation (Score: 6/10)
**Problem**:
- Current docs are good but scattered
- No API reference with examples
- Missing architecture documentation
- Limited migration guides

**Solution**:
- Generate auto-docs from docstrings
- Create API reference guide
- Add architectural decision records (ADRs)
- Create model-by-model training guides
- Add performance benchmarking docs

**Impact**:
- Easier for new contributors
- Better for external users
- Estimated effort: 1-2 weeks

**Tools**: sphinx-autodoc, sphinx-gallery

---

### 🟢 LOW PRIORITY

#### 9. Dependency Management Modernization (Score: 5/10)
**Solution**:
- Migrate setup.py → pyproject.toml (gradual)
- Pin more specific versions
- Create dependency groups (core, dev, extras)
- Add pre-commit hooks (already configured, enforce it)

#### 10. Performance Optimization (Score: 5/10)
**Areas**:
- Data loading pipeline (bottleneck for training)
- GPU memory optimization
- Model inference profiling
- Spectrogram caching strategy

#### 11. Developer Experience (Score: 4/10)
**Improvements**:
- Dev container (Docker + VS Code)
- Makefile improvements
- Development guide for new models
- Auto-formatting on commit

---

## Implementation Phases

### Phase 1: Foundation (Weeks 1-4)
1. Add comprehensive type hints
2. Unify configuration system
3. Consolidate duplicate layers

**Deliverable**: Type-safe, cleaner codebase

### Phase 2: Performance (Weeks 5-8)
1. Modernize to PyTorch 2.0+
2. Add torch.compile() support
3. Optimize data loading

**Deliverable**: 20-40% faster inference, cleaner training

### Phase 3: Flexibility (Weeks 9-12)
1. Decouple from trainer library
2. Add async/streaming inference
3. Support multiple training frameworks

**Deliverable**: More flexible, production-ready API

### Phase 4: Quality (Weeks 13-16)
1. Modernize testing infrastructure
2. Improve documentation
3. Add benchmarking framework

**Deliverable**: Better maintainability, easier for new contributors

---

## Breaking Changes to Plan For

1. **Config Structure**: Old config files won't load directly
   - Migration script needed
   - Backward compatibility layer recommended
   
2. **Trainer API**: If decoupled, custom training scripts need updates
   - Provide migration guide
   - Support old API with deprecation warnings

3. **Layer API**: If consolidated, some imports will change
   - Create import aliases
   - Provide deprecation warnings

4. **Type Hints**: May catch hidden bugs
   - Run type checker in CI
   - Fix real issues discovered

---

## Recommended Reading for Team

1. **Python Dataclasses**: https://docs.python.org/3/library/dataclasses.html
2. **PyTorch 2.0 Features**: https://pytorch.org/blog/pytorch2-0-0-release/
3. **pytest**: https://docs.pytest.org/
4. **Type Hints**: https://peps.python.org/pep-0585/
5. **Architecture Patterns**: "Refactoring: Improving the Design of Existing Code" by Martin Fowler

---

## Success Metrics

After modernization:
- [ ] 100% type hint coverage on public APIs (mypy --strict)
- [ ] 80%+ test coverage (with pytest)
- [ ] 20-40% faster inference (with torch.compile)
- [ ] <10% code duplication (reduced from current 20%+)
- [ ] New model addition takes <1 week (vs 2-3 currently)
- [ ] All examples work with latest PyTorch LTS
- [ ] Full documentation with ADRs

---

## Quick Wins (Can Do Now)

1. Add mypy to CI (non-blocking mode)
2. Consolidate phonemizer implementations
3. Create test coverage report
4. Add torch.compile() to VITS/XTTS for benchmarking
5. Document config inheritance patterns
6. Add performance benchmarking script

---

## Risk Mitigation

- **Risk**: Breaking changes scare users
  - **Mitigation**: Semantic versioning (major bump), deprecation warnings
  
- **Risk**: Performance regressions
  - **Mitigation**: Comprehensive benchmarking, regression testing
  
- **Risk**: Extended migration period
  - **Mitigation**: Parallel implementation (old + new), gradual deprecation
  
- **Risk**: Team bandwidth
  - **Mitigation**: Prioritize high-impact changes, seek community contributions

