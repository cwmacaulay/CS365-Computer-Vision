// Generated by gmmproc 2.47.4 -- DO NOT MODIFY!
#ifndef _GDKMM_VISUAL_P_H
#define _GDKMM_VISUAL_P_H


#include <glibmm/private/object_p.h>

#include <glibmm/class.h>

namespace Gdk
{

class Visual_Class : public Glib::Class
{
public:
#ifndef DOXYGEN_SHOULD_SKIP_THIS
  typedef Visual CppObjectType;
  typedef GdkVisual BaseObjectType;
  typedef GdkVisualClass BaseClassType;
  typedef Glib::Object_Class CppClassParent;
  typedef GObjectClass BaseClassParent;

  friend class Visual;
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

  const Glib::Class& init();


  static void class_init_function(void* g_class, void* class_data);

  static Glib::ObjectBase* wrap_new(GObject*);

protected:

  //Callbacks (default signal handlers):
  //These will call the *_impl member methods, which will then call the existing default signal callbacks, if any.
  //You could prevent the original default signal handlers being called by overriding the *_impl method.

  //Callbacks (virtual functions):
};


} // namespace Gdk


#endif /* _GDKMM_VISUAL_P_H */

